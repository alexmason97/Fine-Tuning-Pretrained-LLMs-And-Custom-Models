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
)

def test_block_quantize_4bit_invalid_dim():
    x2 = torch.randn(2, 5)
    with pytest.raises(RuntimeError):
        block_quantize_4bit(x2.view(-1), group_size=3)

def test_block_quantize_4bit_invalid_group_size_type():
    x = torch.randn(16)
    with pytest.raises(TypeError):
        block_quantize_4bit(x, group_size="8")  # non-int
        
def test_block_quantize_and_dequantize_roundtrip():
    group_size = 8
    x = torch.full((group_size * 2,), 2.0)  
    q4, norm = block_quantize_4bit(x, group_size=group_size)
    assert q4.shape == (2, 4)
    assert norm.shape == (2, 1)
    x_rec = block_dequantize_4bit(q4, norm)
    assert torch.allclose(x_rec, x)
    
def test_quantize_output_ranges_and_types():
    x = torch.linspace(-1, 1, steps=16)
    q4, norm = block_quantize_4bit(x, group_size=8)

    assert q4.dtype == torch.int8
    assert norm.dtype == torch.float16

    lo = (q4 & 0x0F).to(torch.int32)
    hi = ((q4.to(torch.uint8) >> 4) & 0x0F).to(torch.int32)
    assert lo.min() >= 0 and lo.max() <= 15
    assert hi.min() >= 0 and hi.max() <= 15
    
def test_quantize_dynamic_replaces_with_quantized_dynamic_linear():
    model = nn.Sequential(nn.Linear(3,3), nn.ReLU())
    qmodel = quantize_dynamic(model)
    from torch.nn.quantized.dynamic import Linear as DQL     
    assert isinstance(qmodel[0], DQL)