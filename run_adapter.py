import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import TrainingArguments, EarlyStoppingCallback
from adapters import AdapterTrainer
from sklearn.metrics import accuracy_score

from data import load_nli_data
from adapter_model import get_adapter_model

RESULTS_DIR = "results/NLI/Adapter"
SEEDS = [42, 123, 456]
MODEL_NAME = "klue/roberta-base"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def already_done(condition, seed):
    path = os.path.join(RESULTS_DIR, condition.replace("-", "_"), f"seed{seed}", "results.json")
    if os.path.exists(path):
        print(f"[스킵] 이미 완료: {condition} seed{seed}")
        return True
    return False


def save_delta_w(model, condition, seed, output_dir):
    delta_w = {}
    for name, param in model.named_parameters():
        if "adapter" in name.lower() and param.requires_grad:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layer" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in delta_w:
                            delta_w[layer_idx] = 0.0
                        delta_w[layer_idx] += param.norm().item()
                    except ValueError:
                        pass
    if delta_w:
        with open(os.path.join(output_dir, "delta_w.json"), "w") as f:
            json.dump(delta_w, f, indent=2)
        print(f"  Delta W 저장 완료")


def run_cka(condition, seed):
    from adapter_model import get_adapter_model, ADAPTER_BUDGET
    from transformers import AutoModelForSequenceClassification

    print(f"  CKA 분석: {condition} seed{seed}")

    result_dir = os.path.join(RESULTS_DIR, condition.replace("-", "_"), f"seed{seed}")
    cka_path = os.path.join(result_dir, "analysis", "cka_scores.json")
    if os.path.exists(cka_path):
        print(f"  [스킵] CKA 이미 완료: {condition} seed{seed}")
        return

    _, eval_data, _ = load_nli_data("klue/roberta-base")
    subset = torch.utils.data.Subset(eval_data, range(min(500, len(eval_data))))
    loader = torch.utils.data.DataLoader(subset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 베이스 모델 hidden states
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    base_model.eval()

    base_hiddens = [[] for _ in range(12)]
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = base_model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            for i, h in enumerate(outputs.hidden_states[1:]):
                base_hiddens[i].append(h[:, 0, :].cpu().float())

    base_hiddens = [torch.cat(h, dim=0) for h in base_hiddens]
    del base_model
    torch.cuda.empty_cache()

    # 파인튜닝된 Adapter 모델 hidden states
    select4_layers = None
    if condition == "Select-4":
        select4_path = os.path.join(RESULTS_DIR, "select4_layers.json")
        with open(select4_path) as f:
            select4_layers = json.load(f)["layers"]

    ft_model, _ = get_adapter_model(condition, select4_layers=select4_layers)
    ckpt_dir = os.path.join(result_dir, "nli_adapter")
    if os.path.exists(ckpt_dir):
        ft_model.load_adapter(ckpt_dir)
        ft_model.set_active_adapters("nli_adapter")
    ft_model = ft_model.to(device)
    ft_model.eval()

    ft_hiddens = [[] for _ in range(12)]
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = ft_model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            for i, h in enumerate(outputs.hidden_states[1:]):
                ft_hiddens[i].append(h[:, 0, :].cpu().float())

    ft_hiddens = [torch.cat(h, dim=0) for h in ft_hiddens]

    # CKA 계산
    def linear_cka(X, Y):
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        XtX = X.T @ X
        YtY = Y.T @ Y
        XtY = X.T @ Y
        num = (XtY * XtY).sum()
        denom = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())
        return (num / denom).item() if denom > 0 else 0.0

    cka_scores = [linear_cka(base_hiddens[i], ft_hiddens[i]) for i in range(12)]
    mean_cka = float(np.mean(cka_scores))
    most_changed_idx = int(np.argmin(cka_scores))
    least_changed_idx = int(np.argmax(cka_scores))

    result = {
        "task": "NLI", "condition": condition, "seed": seed,
        "cka_scores": cka_scores, "mean_cka": mean_cka,
        "most_changed_layer": most_changed_idx + 1,
        "most_changed_cka": cka_scores[most_changed_idx],
        "least_changed_layer": least_changed_idx + 1,
        "least_changed_cka": cka_scores[least_changed_idx],
    }

    save_dir = Path(result_dir) / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "cka_scores.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [완료] CKA {condition} seed{seed}: mean_cka={mean_cka:.4f}")

    del ft_model
    torch.cuda.empty_cache()


def run_experiment(condition, seed, select4_layers=None):
    if already_done(condition, seed):
        return

    print(f"\n{'='*50}")
    print(f"실험 시작: {condition} | seed={seed}")
    print(f"{'='*50}")

    set_seed(seed)
    train_data, eval_data, _ = load_nli_data("klue/roberta-base")
    model, layers = get_adapter_model(condition, select4_layers=select4_layers)

    output_dir = os.path.join(RESULTS_DIR, condition.replace("-", "_"), f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = AdapterTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 학습 전 초기 파라미터 저장
    init_params = {}
    for name, param in model.named_parameters():
        if "adapter" in name.lower() and param.requires_grad:
            init_params[name] = param.data.clone()

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    eval_result = trainer.evaluate()
    acc = eval_result["eval_accuracy"]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    result = {
        "condition": condition,
        "seed": seed,
        "layers": layers,
        "accuracy": acc,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": trainable / total * 100,
        "elapsed_sec": elapsed,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[완료] {condition} seed{seed}: acc={acc:.4f} | time={elapsed:.1f}s")
    # Delta W 계산 (학습 후 - 학습 전)
    delta_w = {}
    for name, param in model.named_parameters():
        if "adapter" in name.lower() and param.requires_grad and name in init_params:
            diff = (param.data - init_params[name]).norm().item()
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layer" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in delta_w:
                            delta_w[layer_idx] = 0.0
                        delta_w[layer_idx] += diff
                    except ValueError:
                        pass
    if delta_w:
        import json as _json
        with open(os.path.join(output_dir, "delta_w.json"), "w") as f:
            _json.dump(delta_w, f, indent=2)
        print(f"  Delta W 저장 완료")
    del model
    torch.cuda.empty_cache()
    return result


def get_select4_layers():
    layer_scores = {}
    for seed in SEEDS:
        dw_path = os.path.join(RESULTS_DIR, "All_12", f"seed{seed}", "delta_w.json")
        if not os.path.exists(dw_path):
            print(f"[경고] Delta W 없음: seed{seed}")
            continue
        with open(dw_path) as f:
            dw = json.load(f)
        for layer_idx, norm in dw.items():
            layer_idx = int(layer_idx)
            if layer_idx not in layer_scores:
                layer_scores[layer_idx] = []
            layer_scores[layer_idx].append(norm)

    avg_scores = {k: np.mean(v) for k, v in layer_scores.items()}
    sorted_layers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Adapter Delta W Top-12 ===")
    for i, (layer, score) in enumerate(sorted_layers):
        print(f"  Top-{i+1}: Layer {layer} (score={score:.4f})")

    top4 = sorted([layer for layer, _ in sorted_layers[:4]])
    print(f"\nSelect-4 선택 레이어: {top4}")

    select4_path = os.path.join(RESULTS_DIR, "select4_layers.json")
    with open(select4_path, "w") as f:
        json.dump({"layers": top4}, f, indent=2)

    return top4


def summarize():
    conditions = ["All-12", "First-Last-4", "Select-4"]
    print("\n=== 최종 결과 요약 ===")
    for condition in conditions:
        accs = []
        for seed in SEEDS:
            path = os.path.join(RESULTS_DIR, condition.replace("-", "_"), f"seed{seed}", "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    accs.append(json.load(f)["accuracy"])
        if accs:
            print(f"{condition}: mean={np.mean(accs)*100:.2f}% std={np.std(accs)*100:.2f}%")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("="*60)
    print("NLI Adapter 실험 시작")
    print("="*60)

    # Step 1. All-12
    print("\n[Step 1] All-12")
    for seed in SEEDS:
        run_experiment("All-12", seed)

    # Step 2. Select-4 레이어 결정
    print("\n[Step 2] Select-4 레이어 결정")
    select4_layers = get_select4_layers()

    # Step 3. First-Last-4
    print("\n[Step 3] First-Last-4")
    for seed in SEEDS:
        run_experiment("First-Last-4", seed)

    # Step 4. Select-4
    print("\n[Step 4] Select-4")
    for seed in SEEDS:
        run_experiment("Select-4", seed, select4_layers=select4_layers)

    # Step 5. CKA 분석
    print("\n[Step 5] CKA 분석")
    for condition in ["All-12", "First-Last-4", "Select-4"]:
        for seed in SEEDS:
            run_cka(condition, seed)

    # Step 6. 요약
    summarize()

    print("\n모든 실험 완료!")
