# CUDA_VISIBLE_DEVICES=1 python blip2_inference.py \
#     --ckpt-dir ./model/test/best_checkpoint \
#     --eval-data data/test_data/test_data/ \
#     --output results/blipAugTest.csv --cache-dir ~/.cache

python src/ensemble.py \
    --eval-data ../data/test_data/test_data/ \
    --vilt-model ../model/baselineViLT/checkpoint-68000 \
    --vilt-output ../results/vilt.csv \
    --blip-model ../model/BLIPAug2/checkpoint_31 \
    --blip-output ../results/blipAugTest.csv \
    --output-file ../results/ensemble.csv
