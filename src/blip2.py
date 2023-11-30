from PIL import Image
import requests
import torch
# from transformers import BlipProcessor, BlipForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from transformers import Blip2Processor, DefaultDataCollator, TrainingArguments, Trainer, Blip2ForConditionalGeneration

# model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", 
#                                                 cache_dir="/home/congnguyen/drive/.cache/", torch_dtype=torch.float64)
# processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", 
#                                                 cache_dir="/home/congnguyen/drive/.cache/")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch 

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, images, questions, answers, processor):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.processor = processor
        self.max_length = 8
        self.image_height = 128
        self.image_width = 128

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # get image + text
        answers = self.answers[idx]
        questions = self.questions[idx]
        image = Image.open(self.images[idx]).convert("RGB")
        text = self.questions[idx]

        image_encoding = self.processor(image,
                                  # do_resize=True,
                                #   size=(self.image_height,self.image_width),
                                  return_tensors="pt")

        encoding = self.processor(
                                  None,
                                  text,
                                  padding="max_length",
                                  truncation=True,
                                  max_length = self.max_length,
                                  return_tensors="pt"
                                  )
        # # # remove batch dimension
        # encoding = processor(
        #     image, text, padding="max_length", truncation=True, return_tensors="pt"
        # )
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["pixel_values"] = image_encoding["pixel_values"][0]
        # # add labels
        
        # labels = self.processor.tokenizer.encode(
        #     answers,
        #     max_length= 8,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors='pt'
        # )[0]
        labels = self.processor.tokenizer(
            answers,
            padding="max_length",
            truncation=True,
        )
        labels["input_ids"] = [
            [(l if l != self.processor.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        encoding["labels"] = labels["input_ids"]

        return encoding

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = torch.stack(labels)

    return batch

from datasets import load_dataset
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

questions = [item for item in dataset["question"]]
images = [
        f"data/train_fill_in_blank/train_fill_in_blank/{pid}/image.png" for pid in dataset["pid"]
    ] 
answers = [item for item in dataset["answer"]]

dataset = VQADataset(questions = questions,
                          answers = answers,
                          images = images,
                          processor=processor)
# train_set, val_set = torch.utils.data.random_split(dataset, [13549, 1000])

# test_dataset = VQADataset(questions = questions,
#                           answers = answers,
#                           image_paths = images,
#                           processor=processor)

print("Loading data is finished!")
batch_size = 4
train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(val_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=0)

batch = next(iter(train_dataloader))
for k,v in batch.items():
    print(k, v.shape)

from tqdm import tqdm
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

for epoch in range(100):
    print(f"Epoch: {epoch}")
    total_loss = []
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**batch)
        loss = outputs.loss
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss:", sum(total_loss))
    torch.save(model, "./model/BLIP2.2/checkpoint_"+str(epoch))

