import sys
import types
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path

from Fine_Tuning.optimizations.quantization import (
    quantize_dynamic,
    block_quantize_4bit,
    block_dequantize_4bit,
    Linear4Bit,
    quantize_4bit,
)

def test_block_quantize_4bit_invalid_dim():
    x2 = torch.randn(3, 4)
    with pytest.raises(AssertionError):
        block_quantize_4bit(x2, group_size=4)
        
def test_block_quantize_4bit_nonmultiple_length():
    x = torch.randn(10)  # length 10 not divisible by 4
    with pytest.raises(AssertionError):
        block_quantize_4bit(x, group_size=4)
        
def test_block_quantize_and_dequantize_roundtrip():
    # pick a constant nonzero vector so normalization ≠ 0 and quant/dequant roundtrips exactly
    group_size = 8
    x = torch.full((group_size * 2,), 2.0)  # length 16
    q4, norm = block_quantize_4bit(x, group_size=group_size)
    # shapes: (16/8, 8/2) = (2,4)
    assert q4.shape == (2, 4)
    assert norm.shape == (2, 1)
    # dequantize
    x_rec = block_dequantize_4bit(q4, norm)
    # because x was constant and perfectly representable, we get exactly back 2.0
    assert torch.allclose(x_rec, x)
    
def test_block_dequantize_4bit_invalid_dim():
    q = torch.randint(0, 16, (5,))  # 1D
    norm = torch.randn(5, 1)
    with pytest.raises(AssertionError):
        block_dequantize_4bit(q, norm)
    
def test_quantize_dynamic_replaces_with_quantized_dynamic_linear():
    model = nn.Sequential(nn.Linear(3,3), nn.ReLU())
    qmodel = quantize_dynamic(model)
    from torch.nn.quantized.dynamic import Linear as DQL     # dynamic quantized Linear
    # first child should now be a dynamic quantized Linear
    assert isinstance(qmodel[0], DQL)