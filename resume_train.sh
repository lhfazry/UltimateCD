#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python tools/train.py --resume-from $2 $3