# I491E.Visual_Question_Answering
Step1: Setup environment
pip install -r requirement.txt

Step2: Load Data
You must to download dataset from Kaggle links: https://www.kaggle.com/competitions/jaisti491evisualquestionanswering/data
## Getting Started!
# Inference for final submission
python src/ensemble.py \
    --eval-data ../data/test_data/test_data/ \
    --vilt-model ../model/baselineViLT/checkpoint-68000 \
    --vilt-output ../results/vilt.csv \
    --blip-model ../model/BLIPAug2/checkpoint_31 \
    --blip-output results/blipAugTest.csv \
    --output-file ../results/ensemble.csv

# Where:
--eval-data: is path link to testing set
--vilt-model: is path linked to ViLT checkpoint
--vilt-output: is output of ViLT
--blip-model: is path linked to ViLT checkpoint
--blip-output: is output of BLIP model
--output-file

## Fine-tune a ViLT model!
$python hyper_train.py

## Fine-tune a BLIP model!
Step1: Enrich Training data
You can run the notebook $notebook/Augmentation.ipynb

Step2: Fine-tune the BLIP model
$python src/fine_tune.py --model-dir ../model/BLIPAug/ --epoch 100 --cache-dir ~/.cache --train-json data/train_full.jsonl

Step3: Inference the test set
$python src/blip2_inference.py \
    --ckpt-dir ../model/test/best_checkpoint \
    --eval-data ../data/test_data/test_data/ \
    --output ../results/blipAugTest.csv --cache-dir ~/.cache
