# import math
# import pytest
# import torch
# from torch import nn

# from Fine_Tuning.optimizations.qlora import (
#     QLoraConfig,
#     QLoRALinear,
#     apply_qlora,
# )
# from bitsandbytes.nn import Linear4bit as BnbLinear4bit
# from Fine_Tuning.optimizations.quantization import Linear4Bit

# def test_qlora_config_defaults():
#     cfg = QLoraConfig()
#     assert isinstance(cfg.rank, int) and cfg.rank == 4
#     assert isinstance(cfg.alpha, int) and cfg.alpha == 32
#     assert pytest.approx(0.0, abs=1e-6) == cfg.dropout
    
# def test_apply_qlora_replaces_linear_layers():
#     model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
#     q_model = apply_qlora(model, QLoraConfig(rank=2, alpha=2, dropout=0.0))
#     for name, child in q_model.named_children():
#         # each slot “0” and “1” should now be a QLoRALinear
#         assert isinstance(child, QLoRALinear), f"{name=} was not replaced"
    
# def test_qloralinear_init_shapes_and_scaling():
#     cfg = QLoraConfig(rank=8, alpha=16, dropout=0.2)
#     layer = QLoRALinear( in_features=5, out_features=7, config=cfg, bias=True )
#     # correct attributes
#     assert layer.rank == 8
#     assert layer.alpha == 16
#     assert math.isclose(layer.scaling, 16/8)
#     # dropout prob
#     assert isinstance(layer.lora_dropout, nn.Dropout)
#     assert layer.lora_dropout.p == pytest.approx(0.2)
#     # adapter shapes
#     assert tuple(layer.lora_A.shape) == (8, 5)
#     assert tuple(layer.lora_B.shape) == (7, 8)
#     # B starts zero, A nonzero
#     assert torch.all(layer.lora_B == 0)
#     assert not torch.all(layer.lora_A == 0)
#     # base is a quantized‐linear module
#     assert hasattr(layer, "base")
#     # base.weight should exist and be a Tensor of correct shape
#     assert tuple(layer.base.weight.shape) == (7, 5)
#     # if bias=True, base.bias exists
#     assert hasattr(layer.base, "bias") and layer.base.bias is not None
    
    
# def test_apply_qlora_replaces_linears_and_keeps_others():
#     class Sub(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.lin1 = nn.Linear(2,2)
#         def forward(self,x): return self.lin1(x)

#     m = nn.Sequential(Sub(), nn.ReLU(), nn.Linear(2,1))
#     cfg = QLoraConfig(rank=1, alpha=1, dropout=0.0)
#     out = apply_qlora(m, config=cfg)
#     # no pure nn.Linear should remain
#     assert not any(type(x) is nn.Linear for x in out.modules())
#     # but QLoRALinear should appear
#     assert any(isinstance(x, QLoRALinear) for x in out.modules())
#     # other modules untouched
#     assert any(isinstance(x, nn.ReLU) for x in out.modules())
