from .bit_head import BITHead
from .changer import Changer
from .tiny_head import TinyHead
from .swin_head import SwinHead
from .general_scd_head import GeneralSCDHead
from .identity_head import IdentityHead
from .multi_head import MultiHeadDecoder
from .sta_head import STAHead
from .segformer_head import SegformerConvHead, SegFormerMLPHead

__all__ = ['BITHead', 'Changer', 'TinyHead', 'SwinHead', 'GeneralSCDHead',
           'IdentityHead', 'MultiHeadDecoder', 'STAHead', 'SegformerConvHead', 'SegFormerMLPHead']
