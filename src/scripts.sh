tensorboard --logdir model/runs
CUDA_VISIBLE_DEVICES=1 python train.py

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --model-dir ./model/BLIPAug9/ --epoch 100 --cache-dir ~/.cache

CUDA_VISIBLE_DEVICES=1 python blip2_inference.py --ckpt-dir ./model/BLIPAug2/checkpoint_31 --eval-data data/test_data/test_data/ --output results/blipAug31.csv --cache-dir ~/.cache

CUDA_VISIBLE_DEVICES=1 python ensemble.py \
    --vilt-model model/baseline20epoch/checkpoint-68000 --vilt-output results/vilt.csv \
    --eval-data data/test_data/test_data/ \
    --blip-model model/BLIPAug2/checkpoint_31 --blip-output results/blipAug31.csv \
    --output-file "results/ensemble.csv"

python test.py --input results/blipGray.csv --test results/test.csv
