from datasets import load_dataset
from transformers import AutoTokenizer

def load_nli_data(model_name: str, max_seq_length: int = 128):
    """
    KLUE-NLI 데이터셋 로드 + 토크나이징
    반환: tokenized_train, tokenized_val, tokenizer
    """
    print("데이터 로딩 중...")
    dataset = load_dataset("klue", "nli")

    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    print("전처리 중...")
    cols_to_remove = ["guid", "source", "premise", "hypothesis", "label"]

    tokenized_train = dataset["train"].map(preprocess, batched=True, remove_columns=cols_to_remove)
    tokenized_val = dataset["validation"].map(preprocess, batched=True, remove_columns=cols_to_remove)

    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    print(f"train: {len(tokenized_train)}개, val: {len(tokenized_val)}개")
    return tokenized_train, tokenized_val, tokenizer


if __name__ == "__main__":
    train, val, tokenizer = load_nli_data("klue/roberta-base")
    print("input_ids shape:", train[0]["input_ids"].shape)
    print("labels:", train[0]["labels"])