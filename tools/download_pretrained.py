# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import warnings
import gdown
import zipfile
from pathlib import Path
from .utils import create_dir_if_not_exist

PRETRAINED_URLs = {
    'wavevit-s': 'https://drive.google.com/file/d/14ZtFxFiM60Ol5obv2vBHUJVXPfm5VnhE',
    'wavevit-b': 'https://drive.google.com/file/d/1vKA3pJiX5A3KOKNDvZuAXRQFUj4ew9Hu'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Download pretrained utility')
    parser.add_argument(
        '--model-name',
        default='wavevit-b',
        type=str,
        help='Model name')
    parser.add_argument(
        '--output-dir',
        default='./pretrained',
        type=str)
    args = parser.parse_args()
    return args
            
def download_ptretrained(model_name, output_dir):
    create_dir_if_not_exist(output_dir)
    urls = PRETRAINED_URLs[model_name]

    if type(urls) == str:
        gdown(urls, os.path.join(output_dir, f"{model_name}.pth".replace('-', '_')))
            

def main():
    args = parse_args()

    if args.model_name == 'all':
        for ds_name in ['wavevit-s', 'wavevit-b']:
            download_ptretrained(ds_name, args.output_dir)
    else:
        download_ptretrained(args.model_name, args.output_dir)

if __name__ == '__main__':
    main()
