import pytest
import torch
from torch import nn
import torch.nn.functional as F
from Fine_Tuning.optimizations.bfloat16 import BFloat16Linear, apply_bfloat16_torch

def test_requires_grad_flags():
    lin_f32 = nn.Linear(5, 7, bias=True)
    bf16 = BFloat16Linear.from_float(lin_f32)
    assert bf16.weight.dtype == torch.bfloat16
    assert not bf16.weight.requires_grad
    assert bf16.bias is not None
    assert bf16.bias.dtype == torch.bfloat16
    assert not bf16.bias.requires_grad

    lin_nobias = nn.Linear(5, 7, bias=False)
    bf16_nb = BFloat16Linear.from_float(lin_nobias)
    assert bf16_nb.bias is None
    
def test_forward_dtype_roundtrip_and_internal_precision(monkeypatch):
    lin = nn.Linear(3, 4)
    bf16 = BFloat16Linear.from_float(lin)

    x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
    orig_linear = F.linear
    seen = {}
    def capture_linear(input, weight, bias):
        seen['in'], seen['w'], seen['b'] = input.dtype, weight.dtype, (bias.dtype if bias is not None else None)
        return orig_linear(input, weight, bias)
    monkeypatch.setattr(F, "linear", capture_linear)

    out = bf16(x)
    assert seen['in'] == torch.bfloat16
    assert seen['w'] == torch.bfloat16
    assert out.dtype == x.dtype
    
@pytest.mark.parametrize("batch,features", [(1,8), (16, 8), (32,16)])
def test_forward_numerical_close(batch, features):
    lin = nn.Linear(features, features)
    bf16 = BFloat16Linear.from_float(lin)
    x = torch.randn(batch, features)
    out_f32 = lin(x)
    out_bf = bf16(x)
    assert torch.allclose(out_f32, out_bf, atol=5e-2, rtol=5e-2)
 
def test_apply_bfloat16_torch_all_params():
    model = nn.Sequential(nn.Linear(4,4), nn.Conv2d(1,1,1))
    bf_model = apply_bfloat16_torch(model, device="cpu")
    for p in bf_model.parameters():
        assert p.dtype == torch.bfloat16
    for buf in bf_model.buffers():
        assert buf.dtype == torch.bfloat16


def test_no_grad_to_weight_bias():
    lin = nn.Linear(6, 6)
    bf16 = BFloat16Linear.from_float(lin)
    x = torch.randn(4,6, requires_grad=True)
    out = bf16(x).sum()
    out.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert bf16.weight.grad is None
    if bf16.bias is not None:
        assert bf16.bias.grad is None
    

def test_bfloat16linear_from_float_forward_close():
    linear = nn.Linear(4, 3)
    layer = BFloat16Linear.from_float(linear)
    assert layer.weight.dtype == torch.bfloat16
    x = torch.randn(2, 4)
    out_original = linear(x)
    out_layer = layer(x)
    assert torch.allclose(out_original, out_layer, atol=1e-2, rtol=1e-2)
