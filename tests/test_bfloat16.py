import pytest
import torch
from torch import nn
import torch.nn.functional as F
from Fine_Tuning.optimizations.bfloat16 import BFloat16Linear, apply_bfloat16, apply_bfloat16_torch

def test_requires_grad_flags():
    # bias=True
    lin_f32 = nn.Linear(5, 7, bias=True)
    bf16 = BFloat16Linear.from_float(lin_f32)
    # weight/bias are bfloat16 and not trainable
    assert bf16.weight.dtype == torch.bfloat16
    assert not bf16.weight.requires_grad
    assert bf16.bias is not None
    assert bf16.bias.dtype == torch.bfloat16
    assert not bf16.bias.requires_grad

    # bias=False
    lin_nobias = nn.Linear(5, 7, bias=False)
    bf16_nb = BFloat16Linear.from_float(lin_nobias)
    assert bf16_nb.bias is None
    
def test_forward_dtype_roundtrip_and_internal_precision(monkeypatch):
    lin = nn.Linear(3, 4)
    bf16 = BFloat16Linear.from_float(lin)

    # prepare a float32 input
    x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
    # Keep copy of real linear function.
    orig_linear = F.linear
    # monkey-patch F.linear to capture the dtypes seen inside
    seen = {}
    def capture_linear(input, weight, bias):
        seen['in'], seen['w'], seen['b'] = input.dtype, weight.dtype, (bias.dtype if bias is not None else None)
        return orig_linear(input, weight, bias)
    monkeypatch.setattr(F, "linear", capture_linear)

    out = bf16(x)
    # on entry: x should be cast to bfloat16, weight/bias already bfloat16
    assert seen['in'] == torch.bfloat16
    assert seen['w'] == torch.bfloat16
    # output returns to the original dtype
    assert out.dtype == x.dtype
    
@pytest.mark.parametrize("batch,features", [(1,8), (16, 8), (32,16)])
def test_forward_numerical_close(batch, features):
    lin = nn.Linear(features, features)
    bf16 = BFloat16Linear.from_float(lin)
    x = torch.randn(batch, features)
    out_f32 = lin(x)
    out_bf = bf16(x)
    # should be close under a slightly looser tolerance for larger batches
    assert torch.allclose(out_f32, out_bf, atol=5e-2, rtol=5e-2)
 
 # 5. apply_bfloat16_torch should cast entire model to bfloat16
def test_apply_bfloat16_torch_all_params():
    model = nn.Sequential(nn.Linear(4,4), nn.Conv2d(1,1,1))
    bf_model = apply_bfloat16_torch(model, device="cpu")
    # every parameter and buffer should be bfloat16
    for p in bf_model.parameters():
        assert p.dtype == torch.bfloat16
    for buf in bf_model.buffers():
        # e.g. batchnorm running stats can be float32, those will also be cast
        assert buf.dtype == torch.bfloat16


# 6. gradient only flows to inputs, not to weight/bias
def test_no_grad_to_weight_bias():
    lin = nn.Linear(6, 6)
    bf16 = BFloat16Linear.from_float(lin)
    x = torch.randn(4,6, requires_grad=True)
    out = bf16(x).sum()
    out.backward()
    # input grads exist
    assert x.grad is not None and x.grad.shape == x.shape
    # but bf16.weight/bias stay without grad
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
    # element wise compare between tensors w/ tolerance thresholds
    assert torch.allclose(out_original, out_layer, atol=1e-2, rtol=1e-2)
