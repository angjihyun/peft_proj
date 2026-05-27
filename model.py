import random
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def get_model(condition: str, lora_config: dict = None, num_labels: int = 3):
    """
    조건에 따라 모델 반환
    condition: 'FFT', 'All-12', 'First-Last-4', 'Random-4', 'Select-4', 'Select-4-Attention-Only', 'Select-4-FFN-Only'
    """
    print(f"모델 로딩 중... (조건: {condition})")
    model = AutoModelForSequenceClassification.from_pretrained(
        "klue/roberta-base",
        num_labels=num_labels,
    )

    if condition == "FFT":
        return model, None

    layers = get_layers(condition, lora_config)
    rank = lora_config["rank"]
    alpha = lora_config["alpha"]
    target_modules = get_target_modules(condition)

    config = LoraConfig(
    r=rank,
    lora_alpha=alpha,
    lora_dropout=0.1,
    target_modules=target_modules,
    layers_to_transform=layers,
    bias="none",
)

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, layers


def get_layers(condition: str, lora_config: dict):
    if condition == "All-12":
        return list(range(12))
    elif condition == "First-Last-4":
        return [0, 1, 10, 11]
    elif condition == "Random-4":
        layers = random.sample(range(12), 4)
        layers.sort()
        print(f"Random-4 선택 레이어: {layers}")
        return layers
    elif condition in ["Select-4", "Select-4-Attention-Only", "Select-4-FFN-Only"]:
        layers = lora_config.get("layers")
        if layers is None:
            raise ValueError("Select-4 레이어 미확정. All-12 먼저 실행하세요.")
        return layers
    else:
        raise ValueError(f"알 수 없는 조건: {condition}")


def get_target_modules(condition: str):
    if condition == "Select-4-Attention-Only":
        return ["query", "key", "value"]
    elif condition == "Select-4-FFN-Only":
        return ["intermediate.dense", "output.dense"]
    else:
        return ["query", "value"]


if __name__ == "__main__":
    # FFT 테스트
    model, _ = get_model("FFT")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FFT - 전체: {total:,} / 학습가능: {trainable:,} ({trainable/total*100:.4f}%)")

    # All-12 테스트
    model, layers = get_model("All-12", {"rank": 4, "alpha": 8})
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All-12 - 전체: {total:,} / 학습가능: {trainable:,} ({trainable/total*100:.4f}%)")
    print(f"All-12 레이어: {layers}")