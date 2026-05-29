import matplotlib.pyplot as plt
import seaborn as sns

def save_delta_w_heatmap(delta_w_norms, task, condition, seed):
    layer_norms = {}
    for name, norm in delta_w_norms.items():
        for i in range(12):
            if f"layer.{i}." in name:
                layer_norms[f"L{i}"] = layer_norms.get(f"L{i}", 0) + norm

    layers = [f"L{i}" for i in range(12)]
    values = [layer_norms.get(layer, 0) for layer in layers]
    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap([values], annot=True, fmt=".3f", xticklabels=layers, yticklabels=[condition], cmap="YlOrRd", ax=ax)
    plt.tight_layout()
    plt.savefig(f"results/{task}/{condition}/seed{seed}/analysis/delta_w_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()