from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch 
import argparse

parser = argparse.ArgumentParser(
                    prog='Code Llama',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--model-dir', required=True, help='Model Path')
parser.add_argument('--epoch', required=True, help='epoch', default=90)
parser.add_argument('--cache-dir', required=True, help='Model Path')
args = parser.parse_args()

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", 
                                                cache_dir=args.cache_dir, torch_dtype=torch.float64)
# model = torch.load(args.model_dir+"checkpoint_10", map_location=torch.device('cpu'))
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", 
                                                cache_dir=args.cache_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if questions == "What number is shown?":
            image = Image.open(self.images[idx]).convert("L")
        else:
            image = Image.open(self.images[idx]).convert("RGB")
        # image = Image.open(self.images[idx]).convert("L")
        text = self.questions[idx]

        image_encoding = self.processor(image,
                                #   do_resize=True,
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
        
        labels = self.processor.tokenizer.encode(
            answers,
            max_length= self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )[0]
        encoding["labels"] = labels

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
dataset = load_dataset("json", data_files="data/train_full.jsonl", split="train")

questions = [item for item in dataset["question"]]
# images = [
#         f"data/train_fill_in_blank/train_fill_in_blank/{pid}/image.png" for pid in dataset["pid"]
#     ] 
images = []

for pid in dataset["pid"]:
    if "aug" in pid:
        images.append(f"data/augmentation/{pid}/image.png")
    else:
        images.append(f"data/train_fill_in_blank/train_fill_in_blank/{pid}/image.png")

answers = [item for item in dataset["answer"]]

dataset = VQADataset(questions = questions,
                          answers = answers,
                          images = images,
                          processor=processor)
# train_set, val_set = torch.utils.data.random_split(dataset, [13549, 1000])
print("Number of training samples: ", len(answers))
# test_dataset = VQADataset(questions = questions,
#                           answers = answers,
#                           image_paths = images,
#                           processor=processor)

print("Loading data is finished!")
batch_size = 22
train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
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

best_model = 999999999
for epoch in range(int(args.epoch)):
    print(f"Epoch: {epoch}")
    total_loss = []
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        with torch.autocast(device_type='cuda', dtype=torch.float64):
            outputs = model(**batch)
        loss = outputs.loss
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    torch.save(model, args.model_dir + "checkpoint_"+str(epoch))
    print("Loss:", sum(total_loss))
    if best_model > sum(total_loss):
        best_model = sum(total_loss)
        torch.save(model, args.model_dir + "best_checkpoint")

