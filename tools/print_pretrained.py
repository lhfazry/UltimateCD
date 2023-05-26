import torch 
from collections import OrderedDict
import argparse

def print_pretrained(args):
    state_dict = torch.load(args.pretrained)
    print(state_dict)

def main():
    parser = argparse.ArgumentParser(description='Fix pretrained')
    parser.add_argument('pretrained', help='pretrainde file')
    args = parser.parse_args()

    print_pretrained(args)

if __name__ == '__main__':
    main()