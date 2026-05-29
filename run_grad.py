import json, time, random, numpy as np, torch, yaml
from pathlib import Path
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import sys
sys.path.append("/workspace")
from data import load_nli_data
from model import get_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": round(accuracy_score(labels, preds), 6)}

class GradientNormCallback:
    def __init__(self):
        self.layer_norms = {i: [] for i in range(12)}
        self.step = 0
    def on_step(self, model):
        if self.step >= 100: return
        for name, param in model.named_parameters():
            if param.grad is None: continue
            for i in range(12):
                if f"layer.{i}." in name:
                    self.layer_norms[i].append(param.grad.norm().item())
        self.step += 1
    def summary(self):
        return {f"layer_{i}": round(float(np.mean(v)), 6) if v else 0.0 for i, v in self.layer_norms.items()}

class GradNormTrainer(Trainer):
    def __init__(self, *args, grad_norm_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm_cb = grad_norm_callback
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.grad_norm_cb: self.grad_norm_cb.on_step(model)
        return loss

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

def save_grad_heatmap(grad_norms, condition, seed, save_dir):
    layers = [f"L{i}" for i in range(12)]
    values = [grad_norms.get(f"layer_{i}", 0.0) for i in range(12)]
    fig, ax = plt.subplots(figsize=(14, 2))
    sns.heatmap([values], annot=True, fmt=".4f", xticklabels=layers, yticklabels=["Grad Norm"], cmap="Blues", ax=ax)
    plt.title(f"Gradient Norm | NLI / {condition} / seed{seed}")
    plt.tight_layout()
    plt.savefig(save_dir / "gradient_norm_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

with open("/workspace/config.yaml") as f:
    config = yaml.safe_load(f)

conditions = ["All-12", "First-Last-4", "Random-4", "Select-4"]
for condition in conditions:
    for seed in [42, 123, 456]:
        analysis_dir = Path(f"results/NLI/{condition}/seed{seed}/analysis")
        if (analysis_dir / "gradient_norm.json").exists():
            print(f"[스킵] {condition} seed{seed}")
            continue
        print(f"\n=== Grad Norm: {condition} / seed{seed} ===")
        set_seed(seed)
        train_data, val_data, _ = load_nli_data(config["model"]["name"], config["model"]["max_seq_length"])
        lora_cfg = config["lora"].get(condition.lower().replace("-", "_"), {})
        model, _ = get_model(condition, lora_cfg)
        is_fft = condition == "FFT"
        train_cfg = config["training"]["fft"] if is_fft else config["training"]["lora"]
        training_args = TrainingArguments(
            output_dir=f"results/NLI/{condition}/seed{seed}",
            logging_steps=50, eval_strategy="epoch",
            save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=False,
            learning_rate=train_cfg["learning_rate"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=train_cfg["batch_size"],
            num_train_epochs=train_cfg["num_epochs"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            seed=seed, data_seed=seed, full_determinism=True,
            fp16=True, report_to="none",
        )
        grad_cb = GradientNormCallback()
        trainer = GradNormTrainer(model=model, args=training_args,
            train_dataset=train_data, eval_dataset=val_data,
            compute_metrics=compute_metrics, grad_norm_callback=grad_cb)
        trainer.train()
        grad_summary = grad_cb.summary()
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / "gradient_norm.json", "w") as f:
            json.dump(grad_summary, f, indent=2)
        save_grad_heatmap(grad_summary, condition, seed, analysis_dir)
        print(f"완료: {condition} seed{seed}")
        del model; torch.cuda.empty_cache()

print("전체 완료!")
