import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_nli_data
from model import get_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": round(accuracy_score(labels, preds), 6)}


# ── Gradient Norm Hook ──────────────────────────────
class GradientNormCallback:
    def __init__(self):
        self.layer_norms = {i: [] for i in range(12)}
        self.step = 0

    def on_step(self, model):
        if self.step >= 100:
            return
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            for i in range(12):
                if f"layer.{i}." in name:
                    self.layer_norms[i].append(param.grad.norm().item())
        self.step += 1

    def summary(self) -> dict:
        return {
            f"layer_{i}": round(float(np.mean(v)), 6) if v else 0.0
            for i, v in self.layer_norms.items()
        }


class GradNormTrainer(Trainer):
    def __init__(self, *args, grad_norm_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm_cb = grad_norm_callback

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.grad_norm_cb:
            self.grad_norm_cb.on_step(model)
        return loss


# ── Delta W ─────────────────────────────────────────
def compute_delta_w(model):
    delta_w_norms = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            try:
                A = module.lora_A["default"].weight
                B = module.lora_B["default"].weight
                delta_w_norms[name] = torch.norm(B @ A).item()
            except Exception as e:
                print(f"[ΔW 경고] {name}: {e}")
    return delta_w_norms


# ── Accuracy 직접 계산 ────────────────────────────────
def evaluate_accuracy(model, dataset):
    model.eval()
    all_preds, all_labels = [], []
    loader = DataLoader(dataset, batch_size=32)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return round(accuracy_score(all_labels, all_preds), 6)


# ── Heatmap 저장 ──────────────────────────────────────
def save_delta_w_heatmap(delta_w_norms, task, condition, seed):
    layer_norms = {}
    for name, norm in delta_w_norms.items():
        for i in range(12):
            if f"layer.{i}." in name:
                layer_norms[i] = layer_norms.get(i, 0) + norm

    layers = [f"L{i}" for i in range(12)]
    values = [layer_norms.get(i, 0.0) for i in range(12)]

    fig, ax = plt.subplots(figsize=(14, 2))
    sns.heatmap([values], annot=True, fmt=".3f", xticklabels=layers,
                yticklabels=[condition], cmap="YlOrRd", ax=ax, linewidths=0.5)
    plt.title(f"ΔW Norm Heatmap | NLI / {condition} / seed{seed}", fontsize=11)
    plt.tight_layout()

    save_dir = Path(f"results/{task}/{condition}/seed{seed}/analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "delta_w_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ΔW Heatmap] 저장: {path}")


def save_grad_norm_heatmap(grad_norms, task, condition, seed):
    layers = [f"L{i}" for i in range(12)]
    values = [grad_norms.get(f"layer_{i}", 0.0) for i in range(12)]

    fig, ax = plt.subplots(figsize=(14, 2))
    sns.heatmap([values], annot=True, fmt=".4f", xticklabels=layers,
                yticklabels=["Grad Norm"], cmap="Blues", ax=ax, linewidths=0.5)
    plt.title(f"Gradient Norm (first 100 steps) | NLI / {condition} / seed{seed}", fontsize=11)
    plt.tight_layout()

    save_dir = Path(f"results/{task}/{condition}/seed{seed}/analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "gradient_norm_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GradNorm Heatmap] 저장: {path}")


# ── 결과 저장 ─────────────────────────────────────────
def save_results(model, trainer, task, condition, seed, elapsed,
                 val_dataset, grad_summary, selected_layers=None):
    save_dir = Path(f"results/{task}/{condition}/seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = save_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    delta_w_norms = compute_delta_w(model) if condition != "FFT" else {}
    accuracy = evaluate_accuracy(model, val_dataset)
    eval_result = trainer.evaluate()

    results = {
        "task": task,
        "condition": condition,
        "seed": seed,
        "selected_layers": selected_layers,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": round(trainable / total * 100, 4),
        "elapsed_seconds": round(elapsed, 1),
        "accuracy": accuracy,
        "loss": eval_result.get("eval_loss"),
        "gradient_norm": grad_summary,
        "delta_w_norms": delta_w_norms,
        "training_history": trainer.state.log_history,
    }

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(analysis_dir / "gradient_norm.json", "w") as f:
        json.dump(grad_summary, f, indent=2)

    if delta_w_norms:
        with open(analysis_dir / "delta_w_norms.json", "w") as f:
            json.dump(delta_w_norms, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  NLI / {condition} / seed{seed}")
    print(f"  Trainable : {trainable:,} ({trainable/total*100:.4f}%)")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Loss      : {eval_result.get('eval_loss'):.4f}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"{'='*55}\n")
    return results


# ── 실험 실행 ─────────────────────────────────────────
def run_experiment(condition: str, seed: int, config: dict):
    save_dir = Path(f"results/NLI/{condition}/seed{seed}")
    if (save_dir / "results.json").exists():
        print(f"[Skip] {condition} seed{seed} already done")
        return None
    set_seed(seed)
    task = "NLI"
    print(f"\n=== 실험 시작: {condition} / seed {seed} ===")

    train_data, val_data, _ = load_nli_data(
        config["model"]["name"],
        config["model"]["max_seq_length"]
    )

    lora_key = condition.lower().replace("-", "_")
    lora_cfg = config["lora"].get(lora_key, {})
    model, selected_layers = get_model(condition, lora_cfg)

    is_fft = condition == "FFT"
    train_cfg = config["training"]["fft"] if is_fft else config["training"]["lora"]

    output_dir = f"results/{task}/{condition}/seed{seed}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"logs/{task}/{condition}/seed{seed}",
        logging_steps=50,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        num_train_epochs=train_cfg["num_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        seed=seed,
        data_seed=seed,
        full_determinism=True,
        fp16=True,
        report_to="none",
    )

    grad_norm_cb = GradientNormCallback()

    trainer = GradNormTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        grad_norm_callback=grad_norm_cb,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    grad_summary = grad_norm_cb.summary()

    # 히트맵 저장
    if condition != "FFT":
        delta_w = compute_delta_w(model)
        save_delta_w_heatmap(delta_w, task, condition, seed)
    save_grad_norm_heatmap(grad_summary, task, condition, seed)

    results = save_results(model, trainer, task, condition, seed,
                           elapsed, val_data, grad_summary, selected_layers)
    del model
    torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    with open("/workspace/peft_proj/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    conditions = ["QV", "Q-only", "K-only", "V-only", "QKV"]
    task = "NLI"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for condition in conditions:
        for seed in config["seeds"]:
            run_experiment(condition, seed, config)
    import subprocess
    print("\n=== CKA 분석 시작 ===")
    subprocess.run(["python", "run_cka.py"], check=True)

    print("\n전체 완료!")
