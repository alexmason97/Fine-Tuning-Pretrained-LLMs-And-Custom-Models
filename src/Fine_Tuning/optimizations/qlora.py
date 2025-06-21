from dataclasses import dataclass 
from typing import Optional

import math 

import torch 
from torch import nn

# Importing and testing bitsandbytes import for linear quant units 
try:
    from bitsandbytes.nn import Linear4bit 
except ImportError:
    Linear4bit = None 
    
@dataclass
class QLoraConfig:
    
    rank: int = 4
    alpha: int = 32 
    dropout: float = 0.0 
    
class QLoraLinear(nn.Module):
    
    def __init__(
        self, in_features: int, out_features: int, config: QLoraConfig, **kwargs: int
    ):
        super().__init__()
        if Linear4bit is None:
            raise ImportError("bitsandbytes is required for QLoRA")
        self.base = Linear4bit(in_features, out_features, bias=kwargs.get("bias", True))
        self.rank = config.rank 
        self.alpha = config.alpha 
        self.scaling = self.alpha / self.rank 
        self.lora_dropout = nn.Dropout(p=config.dropout)
        self.lora_A = nn.Parameter(torch.zeros((self.rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor):
        forward_pass_result = self.base(x)
        lora_output = self.lora_B @ self.lora_A 
        lora_output = self.lora_dropout(x) @ lora_output.T
        return forward_pass_result + lora_output * self.scaling
    
def apply_qlora(model: nn.Module, config: Optional[QLoraConfig] = None):
    
    if config is None:
        config = QLoraConfig() 
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name, 
                QLoraLinear(
                    module.in_features,
                    module.out_features,
                    config,
                    bias=module.bias is not None,
                ),
            )
        else:
            apply_qlora(module, config)  
    return model 