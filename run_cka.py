import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from peft import PeftModel
import sys
sys.path.append("/workspace")
from data import load_nli_data

def compute_linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    dot_XX = torch.norm(torch.matmul(X, X.T), p='fro')
    dot_YY = torch.norm(torch.matmul(Y, Y.T), p='fro')
    dot_XY = torch.norm(torch.matmul(X, Y.T), p='fro')
    return ((dot_XY ** 2) / (dot_XX * dot_YY)).item()

def get_hidden_states(model, dataloader, device, num_samples=200):
    model.eval()
    all_hidden = [[] for _ in range(13)]
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**batch, output_hidden_states=True)
            for i, hs in enumerate(outputs.hidden_states):
                all_hidden[i].append(hs[:, 0, :].cpu())
            count += next(iter(batch.values())).shape[0]
    return [torch.cat(h, dim=0)[:num_samples] for h in all_hidden]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, val_data, _ = load_nli_data("klue/roberta-base", 128)
loader = DataLoader(val_data, batch_size=32)

base_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=3).to(device)
base_hidden = get_hidden_states(base_model, loader, device)
del base_model
torch.cuda.empty_cache()

conditions = ["QV", "Q-only", "K-only", "V-only", "QKV"]
for condition in conditions:
    for seed in [42, 123, 456]:
        checkpoint_dir = Path(f"results/NLI/{condition}/seed{seed}")
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not checkpoints:
            print(f"[스킵] 체크포인트 없음: {condition} seed{seed}")
            continue
        checkpoint = str(checkpoints[-1])
        if condition == "FFT":
            ft_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3).to(device)
        else:
            ft_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=3)
            ft_model = PeftModel.from_pretrained(ft_model, checkpoint).to(device)
        ft_hidden = get_hidden_states(ft_model, loader, device)
        cka_scores = [round(compute_linear_cka(base_hidden[i], ft_hidden[i]), 4) for i in range(1, 13)]
        mean_cka = round(float(np.mean(cka_scores)), 4)
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
        save_dir = Path(f"results/NLI/{condition}/seed{seed}/analysis")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "cka_scores.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"[완료] {condition} seed{seed}: mean_cka={mean_cka:.4f}")
        del ft_model
        torch.cuda.empty_cache()

print("CKA 분석 완료!")
