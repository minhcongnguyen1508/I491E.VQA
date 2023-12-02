# I491E.Visual_Question_Answering
Step 1: Setup environment
```console
pip install -r requirement.txt
```
Step 2: Load Data
You must to download data from Kaggle links: https://www.kaggle.com/competitions/jaisti491evisualquestionanswering/data. 

Step 3: Download the fine-tuned model

# Getting Started!
## Inference for final submission
```console
python src/ensemble.py \
    --eval-data ./data/test_data/test_data/ \
    --vilt-model ./model/baselineViLT/checkpoint-68000 \
    --vilt-output ./results/vilt.csv \
    --blip-model ./model/BLIPAug2/checkpoint_31 \
    --blip-output ./results/blipAugTest.csv \
    --output-file ./results/ensemble.csv
    --cache-dir ~/.cache
```

## Where:
```console
--eval-data: is path link to testing set
--vilt-model: is path linked to ViLT checkpoint
--vilt-output: is output of ViLT
--blip-model: is path linked to ViLT checkpoint
--blip-output: is output of BLIP model
--output-file
```

### Fine-tune a new ViLT model!
```console
$python hyper_train.py
```

### Fine-tune a new BLIP model!
Step 1: Enrich Training data
```console
You can run the notebook $notebook/Augmentation.ipynb
```

Step 2: Fine-tune the BLIP model
```console
$python src/fine_tune.py --model-dir ./model/BLIPAug/ --epoch 100 --cache-dir ~/.cache --train-json data/train_full.jsonl
```

Step 3: Inference the test set
```console
$python src/blip2_inference.py \
    --ckpt-dir ./model/BLIP/best_checkpoint \
    --eval-data ./data/test_data/test_data/ \
    --output ./results/blipAugTest.csv --cache-dir ~/.cache
```

If have any issues, please feel free contact to me via email at congnhm@jaist.ac.jp!
