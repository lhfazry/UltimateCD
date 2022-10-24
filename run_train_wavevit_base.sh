#!/bin/bash
mkdir -p lightning_logs/wavevit

CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py \
    --data_dir=datasets/cropped \
    --pretrained=pretrained/wavevit_b.pth.tar \
    --batch_size=2 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=small \
    --max_epoch=15 \
    --sampling_strategy=truncate \
    --csv_file=datasets/video_exposure.csv