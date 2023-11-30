from datasets import load_dataset
from transformers import (
    ViltProcessor,
    DefaultDataCollator,
    ViltForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")


answers = [item for item in dataset["answer"]]
unique_labels = list(set(answers))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}


# model = "dandelin/vilt-b32-finetuned-vqa"
# processor = ViltProcessor.from_pretrained(model)
model = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model)

def preprocess_data(examples):
    pids = examples["pid"]

    image_paths = [
        f"data/train_fill_in_blank/train_fill_in_blank/{pid}/image.png" for pid in pids
    ]

    images = [Image.open(image_path) for image_path in image_paths]
    texts = examples["question"]

    try:
        encoding = processor(
            images, texts, padding="max_length", truncation=True, return_tensors="pt"
        )
    except Exception as e:
        print(f"Error {e} in processor , will skip this batch")
        return {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "pixel_mask": [],
        }

    for k, v in encoding.items():
        encoding[k] = v.squeeze()

    targets = []

    for answer in examples["answer"]:
        target = torch.zeros(len(id2label))
        answer_id = label2id[answer]
        target[answer_id] = 1.0
        targets.append(target)

    encoding["labels"] = targets
    return encoding


processed_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=[
        "question",
        "answer",
        "ques_type",
        "grade",
        "label",
        "pid",
        "hint",
        "unit",
    ],
)


data_collator = DefaultDataCollator()
# model = ViltForQuestionAnswering.from_pretrained(
#     model,
#     num_labels=len(id2label),
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,
# )

model = ViTForImageClassification.from_pretrained(
    model,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="model/ViT32k",
    per_device_train_batch_size=64,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=200,
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    save_strategy='steps',
    evaluation_strategy="steps",
    report_to="tensorboard",
)

split_ratio = 0.9
split_idx = round(len(processed_dataset) * split_ratio)
train_ds = processed_dataset.select(range(split_idx))
eval_ds = processed_dataset.select(range(split_idx, len(processed_dataset)))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=processor,
)


trainer.train()
