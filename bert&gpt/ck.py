from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 1. 加载数据（示例使用本地TSV文件）
dataset = load_dataset("csv",
                      data_files={
                          "train": "train.tsv",
                          "test": "test.tsv"
                      },
                      delimiter="\t",
                      column_names=["label", "text"])

# 2. 预处理
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def preprocess(examples):
    labels = [int(label) for label in examples["label"]]
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized["labels"] = labels
    return tokenized

tokenized_ds = dataset.map(preprocess, batched=True)

# 3. 训练
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
)

trainer.train()