from typing import NamedTuple
import torch.nn as nn

class Default(NamedTuple):
    num_worker: int = 4
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.05
    image_size: int = 32
    patch_size: int = 4
    dim: int = 4*4*3*2*2
    num_heads: int = 6
    activation = nn.GELU()
    qkv_bias: bool = True
    quiet_attention: bool = True
    num_blocks: int = 8
    dropout: float = 0.1
    decoder_num_blocks: int = 6
    decoder_num_heads: int = 6
    mask_ratio: float = 0.5
    pretrain_batch_size: int = 1024
    pretrain_epochs: int = 800
    pretrain_lr: float = 3e-4

    