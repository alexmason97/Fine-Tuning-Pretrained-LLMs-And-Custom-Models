import math
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from Fine_Tuning.optimizations.lora import (
    LoraConfig,
    LoraLinear,
    LoraBFloat16Linear,
    apply_lora,
)

def test_lora_config_defaults():
    cfg = LoraConfig()
    assert isinstance(cfg.rank, int) and cfg.rank == 32
    assert isinstance(cfg.alpha, int) and cfg.alpha == 16
    assert pytest.approx(0.05, rel=1e-6) == cfg.dropout
    
def test_loralinear_grad_flags():
    cfg = LoraConfig(rank=4, alpha=8, dropout=0.1)
    layer = LoraLinear( in_features=5, out_features=7, config=cfg )
    # base Linear weight/bias should be trainable
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad
    # LoRA adapters should also be trainable
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad
    
    
def test_lora_linear_forward_with_manual_lora():
    # make a tiny deterministic example
    # so we can predict the LoRA term exactly
    cfg = LoraConfig(rank=1, alpha=2, dropout=0.0)
    lora = LoraLinear( in_features=3, out_features=2, config=cfg )
    # zero out the base weight & bias
    with torch.no_grad():
        lora.weight.zero_()
        lora.bias.zero_()
        # set A = [1,1,1], B = [[1],[1]]
        lora.lora_A.fill_(1.0)
        lora.lora_B.fill_(1.0)
    # scaling = alpha/rank = 2
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, -1.0]])
    out = lora(x)
    # compute expected: LoRA matrix = B @ A = [[1,1,1],[1,1,1]]
    # so each row of output = x @ [1,1,1].T * 2 = sum(x)*2
    sums = x.sum(dim=1, keepdim=True) * 2  # shape (2,1)
    expected = torch.cat([sums, sums], dim=1)
    assert torch.allclose(out, expected)
