from datasets import load_dataset
from transformers import Blip2Processor, DefaultDataCollator, TrainingArguments, Trainer, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
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

# def load_model(model=None):
#     # "model/baseline/checkpoint-1000"
#     # "model/baseline/checkpoint-1000"
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = torch.load(model, map_location='cuda:0')
#     return processor, model

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
    text_url = eval_data[idx][1]
    textf = open(text_url)
    # returns JSON object as 
    # a dictionary
    data = json.load(textf)
    text = data["question"]
    if text == "What number is shown?":
        image = Image.open(img_url).convert("L")
    else:
        image = Image.open(img_url).convert("RGB")
    return image, text

# # prepare inputs
def inference(model, eval_data, combine=False):
    result = {}
    for id in eval_data:
        image, text = get_data(id)
        encoding = processor(image, text, return_tensors="pt")
        # forward pass
        encoding.to(device)
        # forward pass
        # outputs = model(**encoding)
        out = model.generate(**encoding)
        # logits = outputs.logits
        # idx = logits.argmax(-1).item()
        out = processor.decode(out[0], skip_special_tokens=True).replace(" ","")
        print(out)
        result.update({id: out})
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
    parser.add_argument('--model', required=False, default="BLIP")
    parser.add_argument('--cache-dir', required=False, default="/home/congnguyen/drive/.cache/")
    args = parser.parse_args()
    # print(args.ckpt_dir, args.output)
    # processor, model = load_model(args.ckpt_dir)
    if args.model == "BLIP":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", cache_dir=args.cache_dir)
        model = torch.load(args.ckpt_dir, map_location='cpu')
    else:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir="/home/congnguyen/.cache")
        model = torch.load(args.ckpt_dir)
    model.to(device)
    # processor.to(device)
    model.eval()
    eval_data = load_dataset(args.eval_data)
    print(len(eval_data))
    result = inference(model, eval_data)
    # result = inference(model, eval_data, True)
    write2csv(result, args.output)
