from datasets import load_dataset
from transformers import (
    ViltProcessor,
    DefaultDataCollator,
    ViltForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import torch
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import json
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

def load_model(model=None):
    # "model/baseline/checkpoint-1000"
    # "model/baseline/checkpoint-1000"
    processor = ViltProcessor.from_pretrained(model)
    model = ViltForQuestionAnswering.from_pretrained(model)
    print("Finish!")
    return processor, model

# prepare image + question
def load_dataset(path):
    # "data/test_data/test_data/"
    eval_dir = os.path.abspath(path)
    eval_data = {}
    for it in listdir(eval_dir):
        eval_data.update({it: []})
        for f in listdir(join(eval_dir, it)):
            file = join(eval_dir, it, f)
            eval_data[it].append(file)
    print("Loading data finished!")
    return eval_data

def get_data(idx):
    img_url = eval_data[idx][0]
    image = Image.open(img_url).convert('RGB')
    text_url = eval_data[idx][1]
    textf = open(text_url)
    # returns JSON object as 
    # a dictionary
    data = json.load(textf)
    text = data["question"]
    return image, text

# # prepare inputs
def inference(model, eval_data):
    result = {}
    for id in eval_data:
        image, text = get_data(id)
        encoding = processor(image, text, return_tensors="pt")
        # forward pass
        encoding.to(device)
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        result.update({id: model.config.id2label[idx]})
        # print(idx, " - ",text, " Answer: ",  model.config.id2label[idx])  
    return result

def write2csv(result, output='result.csv'):
    import csv
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for idx in result:
            writer.writerow({'ID': idx, 'Label': result[idx]})

if __name__=="__main__":
    parser = argparse.ArgumentParser(
                    prog='VQA',
                    epilog='Text at the bottom of help')
    parser.add_argument('--ckpt-dir', required=True)      # option that takes a value
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    # print(args.ckpt_dir, args.output)
    processor, model = load_model(args.ckpt_dir)
    model.to(device)
    # processor.to(device)
    eval_data = load_dataset(args.eval_data)
    result = inference(model, eval_data)
    write2csv(result, args.output)
