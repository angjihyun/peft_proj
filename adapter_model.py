import adapters
from adapters import AutoAdapterModel

MODEL_NAME = "klue/roberta-base"
NUM_LABELS = 3

ADAPTER_BUDGET = {
    "All-12":       {"reduction_factor": 19, "layers": list(range(12))},
    "First-Last-4": {"reduction_factor": 6,  "layers": [0, 1, 10, 11]},
    "Select-4":     {"reduction_factor": 6,  "layers": None},
}


def get_adapter_model(condition: str, select4_layers: list = None):
    print(f"[Adapter] 모델 로딩 중... (조건: {condition})")

    model = AutoAdapterModel.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
    )

    cfg = ADAPTER_BUDGET[condition].copy()

    if condition == "Select-4":
        if select4_layers is None:
            raise ValueError("Select-4: All-12 Delta W 결과 후 레이어를 지정해야 합니다.")
        cfg["layers"] = select4_layers

    layers = cfg["layers"]
    rf = cfg["reduction_factor"]
    leave_out = [i for i in range(12) if i not in layers]

    adapter_config = adapters.SeqBnConfig(
        reduction_factor=rf,
        leave_out=leave_out,
    )

    model.add_adapter("nli_adapter", config=adapter_config)
    model.add_classification_head("nli_adapter", num_labels=NUM_LABELS)
    model.train_adapter("nli_adapter")
    model.set_active_adapters("nli_adapter")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {trainable/total*100:.4f}")

    return model, layers
