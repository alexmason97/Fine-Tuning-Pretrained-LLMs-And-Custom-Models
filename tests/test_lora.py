import pytest
import torch
from torch import nn
import torch.nn.functional as F

from Fine_Tuning.optimizations.lora import (
    LoraConfig,
    LoraLinear,
)

def test_lora_config_defaults():
    cfg = LoraConfig()
    assert isinstance(cfg.rank, int) and cfg.rank == 32
    assert isinstance(cfg.alpha, int) and cfg.alpha == 16
    assert pytest.approx(0.05, rel=1e-6) == cfg.dropout
    
def test_loralinear_grad_flags():
    cfg = LoraConfig(rank=4, alpha=8, dropout=0.1)
    layer = LoraLinear( in_features=5, out_features=7, config=cfg )
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad
    
    
def test_lora_linear_forward_with_manual_lora():
    cfg = LoraConfig(rank=1, alpha=2, dropout=0.0)
    lora = LoraLinear( in_features=3, out_features=2, config=cfg )
    with torch.no_grad():
        lora.weight.zero_()
        lora.bias.zero_()
        # set A = [1,1,1], B = [[1],[1]]
        lora.lora_A.fill_(1.0)
        lora.lora_B.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, -1.0]])
    out = lora(x)
    sums = x.sum(dim=1, keepdim=True) * 2  
    expected = torch.cat([sums, sums], dim=1)
    assert torch.allclose(out, expected)
