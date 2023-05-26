import torch 
from collections import OrderedDict
import argparse

def print_pretrained(args):
    
    state_dict = torch.load(args.pretrained)
    
    if 'state_dict' in state_dict:
        for k, v in state_dict['state_dict'].items():
            print(f"{k} ==> {v.shape}")
    else:
        for k, v in state_dict.items():
            print(f"{k} ==> {v.shape}")

    

def main():
    parser = argparse.ArgumentParser(description='Fix pretrained')
    parser.add_argument('pretrained', help='pretrainde file')
    args = parser.parse_args()

    print_pretrained(args)

if __name__ == '__main__':
    main()