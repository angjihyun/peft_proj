import random
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def get_model(condition: str, lora_config: dict = None, num_labels: int = 3):
    print(f"모델 로딩 중... (조건: {condition})")
    model = AutoModelForSequenceClassification.from_pretrained(
        "klue/roberta-base",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
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
        modules_to_save=[],
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
    elif condition in ["QV", "Q-only", "K-only", "V-only", "QKV"]:
        return [0, 1, 10, 11]
    else:
        raise ValueError(f"알 수 없는 조건: {condition}")


def get_target_modules(condition: str):
    if condition == "Select-4-Attention-Only":
        return ["query", "key", "value"]
    elif condition == "Select-4-FFN-Only":
        return ["intermediate.dense", "output.dense"]
    elif condition == "QV":
        return ["query", "value"]
    elif condition == "Q-only":
        return ["query"]
    elif condition == "K-only":
        return ["key"]
    elif condition == "V-only":
        return ["value"]
    elif condition == "QKV":
        return ["query", "key", "value"]
    else:
        return ["query", "value"]
