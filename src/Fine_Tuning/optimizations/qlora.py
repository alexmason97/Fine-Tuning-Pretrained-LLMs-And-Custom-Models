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
    
from .quantization import Linear4Bit, quantize_bnb_4bit
    
@dataclass
class QLoraConfig:
    
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.05
    
class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

    
    @classmethod
    def from_float(cls, module: nn.Linear, config: QLoraConfig):
        layer = cls(
            module.in_features,
            module.out_features,
            config,
            bias=module.bias is not None,
        )
        if hasattr(layer.base.__class__, "from_float"):
            layer.base = layer.base.__class__.from_float(module)
        else:
            layer.base.weight.data.copy_(module.weight.data)
            if module.bias is not None and hasattr(layer.base, "bias"):
                layer.base.bias.data.copy_(module.bias.data)
        return layer   
          
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora_b(self.lora_a(x))
        
    
def apply_qlora(model: nn.Module, config: Optional[QLoraConfig] = None):
    
    if config is None:
        config = QLoraConfig() 
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QLoRALinear.from_float(module, config))
        else:
            apply_qlora(module, config)  
    return model 

def apply_qlora_peft(model: nn.Module, config: Optional[QLoraConfig] = None):
    """Apply QLoRA using the reference `peft` library."""

    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    if config is None:
        config = QLoraConfig()

    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)