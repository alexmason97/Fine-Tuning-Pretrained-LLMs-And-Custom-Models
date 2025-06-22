from dataclasses import dataclass
from typing import Optional

import torch 
import torch.nn.functional as F 
from torch import nn 

def quantize_dynamic(model: nn.Module):
    
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

def block_quantize_4bit(
    x: torch.Tensor, group_size: int = 16
):
    assert x.dim() == 1 
    assert x.size(0) % group_size == 0 
    
    x = x.view(-1, group_size) 
    normalization = x.abs().max(dim=-1, keepdim=True).values 
    x_norm = (x + normalization) / (2 * normalization) 
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)

def block_dequantize_4bit(
    x_quant_4: torch.Tensor, normalization: torch.Tensor
):
    assert x_quant_4.dim() == 2
    
    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15 
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)

class Linear4Bit(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16):
        super().__init__()
        self._shape = (out_features, in_features)
        self.group_size = group_size
        
        self.register_buffer(
            "weight_q4",
            torch.zeros(
                out_features * in_features // group_size,
                group_size // 2,
                dtype=torch.int8,
            ),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(
                out_features * in_features // group_size, 1, dtype=torch.float16
            ),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        
    def _load_state_dict_pre_hook(self, state_dict, prefix):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            flattened_weight = weight.view(-1)
            bit4_q, max_v = block_quantize_4bit(flattened_weight, self._group_size)
            self.weight_q4.copy_(bit4_q)
            self.weight_norm.copy_(max_v)
            
    @classmethod 
    def from_float(cls, module: nn.Linear, group_size: int = 16):
        layer = cls(
            module.in_features, module.out_features, module.bias is not None, group_size
            )
        if module.bias is not None: 
            layer.bias.data.copy_(module.bias.data) 
        weight_flat = module.weight.data.view(-1)
        q4, norm = block_quantize_4bit(weight_flat, group_size)
        layer.weight_q4.copy_(q4)
        layer.weight_norm.copy_(norm)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flat = block_dequantize_4bit(self.weight_q4, self.weight_norm)
            W_vector = flat.view(self._shape)
            return F.linear(x, W_vector, self.bias)
    
def quantize_4bit(model: nn.Module, group_size: int = 16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, Linear4Bit.from_float(module, group_size))
        else:
            quantize_4bit(module, group_size)
    return model

def quantize_bnb_4bit(model: nn.Module):
    try:
        from bitsandbytes.nn import Linear4bit as BnbLinear4bit
    except ImportError as e:
        raise ImportError("We need bitsandbytes for 4-bit quantization. It is currently unavailable") from e

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if hasattr(BnbLinear4bit, "from_float"):
                setattr(model, name, BnbLinear4bit.from_float(module))
            else:
                layer = BnbLinear4bit(module.in_features, module.out_features, bias=module.bias is not None)
                layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    layer.bias.data.copy_(module.bias.data)
                setattr(model, name, layer)
        else:
            quantize_bnb_4bit(module)
    return model