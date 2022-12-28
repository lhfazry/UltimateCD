import torch 
from collections import OrderedDict
import argparse

def fix_pretrained(args):
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.pretrained)
    
    for k, v in state_dict['state_dict'].items():
        k = k.replace('backbone.', '')   # remove prefix backbone.
        #k = k.replace('attn.qkv.weight', 'attn.attn.in_proj_weight')
        #k = k.replace('attn.qkv.bias', 'attn.attn.in_proj_bias')
        #k = k.replace('attn.proj.weight', 'attn.attn.out_proj.weight')
        #k = k.replace('attn.proj.bias', 'attn.attn.out_proj.bias')
        new_state_dict[k] = v

    new_dict = OrderedDict()
    new_dict['state_dict'] = new_state_dict
    torch.save(new_dict, 'pretrained/swin_base_patch244_window877_kinetics400_22k_fixed.pth')

def main():
    parser = argparse.ArgumentParser(description='Fix pretrained')
    parser.add_argument('pretrained', help='pretrainde file')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite')
    args = parser.parse_args()

    fix_pretrained(args)

if __name__ == '__main__':
    main()