# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import warnings
import gdown
import zipfile
from pathlib import Path
from utils import create_dir_if_not_exist

DS_URLs = {
    'levir-cd': {
            "train": "18GuoCuBn48oZKAlEo-LrNwABrFhVALU-",
            "val": "1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG",
            "test": "1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0"
        },
    'dsifn-cd': "10BuTzKyuInzeau2Wx1PRW3Inbply5uQn"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Download dataset utility')
    parser.add_argument(
        '--ds-name',
        default='levir-cd', #all, levir-cd, dsifn-cd
        type=str,
        help='Dataset name')
    parser.add_argument(
        '--extract',
        default=True,
        action='store_true',
        help='Extract the dataset')
    parser.add_argument(
        '--delete-after-extract',
        default=True,
        action='store_true',
        help='Delete the zip file after extract')
    parser.add_argument(
        '--output-dir',
        default='./datasets',
        type=str)
    args = parser.parse_args()
    return args

def download_zip(zip_url, output_dir, split=None, extract=True, delete_after_extract=True):
    #print(zip_url)
    zip_file = os.path.join(output_dir, "data.zip" if split is None else f"{split}.zip")
    gdown.download(id=zip_url, output=zip_file)

    if extract:
        with zipfile.ZipFile(zip_file, 'r') as zip:
            zip.extractall(os.path.join(output_dir) 
                if split is None else os.path.join(output_dir, f"{split}"))

        #delete zip file after extract
        if delete_after_extract:
            os.remove(zip_file)
            
def download_dataset(ds_name, output_dir, extract, delete_after_extract):
    create_dir_if_not_exist(output_dir)
    urls = DS_URLs[ds_name]
    #print(urls)
    
    output_dir = os.path.join(output_dir, ds_name)
    create_dir_if_not_exist(output_dir)

    if type(urls) == dict:
        for split in urls:
            download_zip(urls[split], output_dir, split=split)
    elif type(urls) == str:
        download_zip(urls, output_dir)
            

def main():
    args = parse_args()

    if args.ds_name == 'all':
        for ds_name in ['levir-cd', 'dsifn-cd']:
            download_dataset(ds_name, args.output_dir, args.extract, args.delete_after_extract)
    else:
        download_dataset(args.ds_name, args.output_dir, args.extract, args.delete_after_extract)

if __name__ == '__main__':
    main()
