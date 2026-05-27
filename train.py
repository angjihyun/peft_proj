import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import yaml

from data import load_nli_data
from model import get_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def compute_delta_w(model):
    delta_w_norms = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            delta_w_norms[name] = torch.norm(B @ A).item()
    return delta_w_norms


def save_results(model, trainer, task, condition, seed, elapsed, selected_layers=None):
    save_dir = Path(f"results/{task}/{condition}/seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    delta_w_norms = compute_delta_w(model) if condition != "FFT" else {}

    eval_result = trainer.evaluate()
    print("eval_result:", eval_result)

    results = {
        "task": task,
        "condition": condition,
        "seed": seed,
        "selected_layers": selected_layers,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": round(trainable / total * 100, 4),
        "elapsed_seconds": round(elapsed, 1),
        "accuracy": eval_result.get("eval_accuracy"),
        "loss": eval_result.get("eval_loss"),
        "delta_w_norms": delta_w_norms,
        "training_history": trainer.state.log_history,
    }

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"저장 완료: {save_dir}/results.json")
    return results


def run_experiment(condition: str, seed: int, config: dict):
    set_seed(seed)

    task = "NLI"
    print(f"\n=== 실험 시작: {condition} / seed {seed} ===")

    # 데이터 로드
    train_data, val_data, tokenizer = load_nli_data(
        config["model"]["name"],
        config["model"]["max_seq_length"]
    )

    # 모델 로드
    lora_cfg = config["lora"].get(condition.lower().replace("-", "_"), {})
    model, selected_layers = get_model(condition, lora_cfg)

    # 학습 설정
    is_fft = condition == "FFT"
    train_cfg = config["training"]["fft"] if is_fft else config["training"]["lora"]

    output_dir = f"results/{task}/{condition}/seed{seed}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f"logs/{task}/{condition}/seed{seed}",
        logging_steps=50,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        load_best_model_at_end=False,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    # 학습
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # 결과 저장
    results = save_results(model, trainer, task, condition, seed, elapsed, selected_layers)
    print(f"Accuracy: {results['accuracy']:.4f} | 시간: {results['elapsed_seconds']}s")
    return results


if __name__ == "__main__":
    with open("/workspace/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for seed in config["seeds"]:
        run_experiment("All-12", seed, config)