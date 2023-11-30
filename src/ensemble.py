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
import csv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prepare image + question
def read_evaldata(path="data/test_data/test_data/"):
    eval_dir = os.path.abspath(path)
    eval_data = {}
    for it in listdir(eval_dir):
        eval_data.update({it: []})
        for f in listdir(join(eval_dir, it)):
            file = join(eval_dir, it, f)
            eval_data[it].append(file)
    return eval_data

def get_data(idx):
    img_url = eval_data[idx][0]
    image = Image.open(img_url)
    text_url = eval_data[idx][1]
    textf = open(text_url)
    # returns JSON object as 
    # a dictionary
    data = json.load(textf)
    text = data["question"]
    return image, text

def read_csv(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    results = {}
    for row in csvreader:
        results.update({row[0]: row[1]}) 
    return results

def read_csv2(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    results = {}
    for row in csvreader:
        results.update({row[0]: [row[1],row[2]]}) 
    return results

def write2csv(result, output='./results/ensemble.csv'):
    import csv
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for idx in result:
            writer.writerow({'ID': idx, 'Label': result[idx]})

def vilt(eval_data, model_path="model/baseline20epoch/checkpoint-68000", output='results/vilt.csv'):
    processor = ViltProcessor.from_pretrained(model_path)
    model = ViltForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    result = {}
    for id in eval_data:
        image, text = get_data(id)
        encoding = processor(image, text, return_tensors="pt")
        # forward pass
        encoding.to(device)
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("===========================")
        print(text)
        print(model.config.id2label[idx])
        result.update({id: [model.config.id2label[idx], logits[0][idx].item()]})

    if len(result) == 0:
        print("Can't predict the Vilt model")
        return result

    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Label', 'conf_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in result:
            writer.writerow({'ID': idx, 'Label': result[idx][0], 'conf_score': result[idx][1]})

    return result

def blip(eval_data, ch="model/baseline20epoch/checkpoint-68000", output='results/blipAug2.csv', cache_dir="/home/congnguyen/drive/"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", cache_dir=cache_dir)
    model = torch.load(args.ckpt_dir, map_location='cpu')
    model.to(device)

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
    write2csv(result, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='VQA',
                    epilog='Text at the bottom of help')
    parser.add_argument('--vilt-model', required=True)      # option that takes a value
    parser.add_argument('--blip-model', required=True)
    parser.add_argument('--vilt-output', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--blip-output', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--cache-dir', required=False, default="/home/congnguyen/drive/")
    args = parser.parse_args()
    eval_data = read_evaldata(args.eval_data)
    vilt_result = read_csv2(args.vilt_output)
    blipAug2_result = read_csv(args.blip_output)

    print("Loading data: Done!")
    if len(vilt_result) == 0:
        print("Can't load the vilt result!!!!")
        vilt_result = vilt(eval_data, args.vilt_model, args.vilt_output)
    if len(blipAug2_result) == 0:
        print("Can't load the blip result!!!!")
        blipAug2_result = blip(eval_data, args.blip_model, args.blip_output)

    ensemble_result = {}
    for id in eval_data:
        image, text = get_data(id)
        if text == "What number is shown?":
            ensemble_result.update({id: blipAug2_result[id]})
            # print(id, ": ", blipAug2[id])
        # elif text == "How many shapes are green?":
        #     ensemble_result.update({id: vilt[id]})
        else:
            if float(vilt_result[id][1]) >= 12.5:
                ensemble_result.update({id: vilt_result[id][0]})
            else:
                ensemble_result.update({id: blipAug2_result[id]})
    if len(ensemble_result) != 0:
        write2csv(ensemble_result, args.output_file)
        print("Finish!!!!!!!!!!!")

