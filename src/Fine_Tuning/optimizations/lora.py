from dataclasses import dataclass
from typing import Optional, Sequence

import math 

import torch 
from torch import nn 
from .bfloat16 import BFloat16Linear
@dataclass
class LoraConfig:
    
    rank: int = 32  
    alpha: int = 16
    dropout: float = 0.05 
    
class LoraLinear(nn.Linear):
    
    def __init__(
        self, in_features: int, out_features: int, config: LoraConfig, **kwargs: int
        ):
        super().__init__(in_features, out_features)
        self.rank = config.rank
        self.alpha = config.alpha 
        self.scaling = self.alpha / self.rank 
        self.lora_dropout = nn.Dropout(p=config.dropout)
        self.lora_A = nn.Parameter(torch.zeros((self.rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.rank)))
        # NOTE TO SELF: needed to initialize dist with sqrt(5) to match PyTorch init w/ weights  
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor):
        forward_pass_result = super().forward(x)
        if self.rank > 0:
            lora_output = self.lora_B @ self.lora_A
            lora_output = self.lora_dropout(x) @ lora_output.T 
            forward_pass_result = forward_pass_result + lora_output * self.scaling
        return forward_pass_result
    
class LoraBFloat16Linear(BFloat16Linear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)        
        torch.nn.init.zeros_(self.lora_b.weight)

    @classmethod
    def from_float(cls, module: nn.Linear, config: LoraConfig):
        layer = cls(module.in_features, module.out_features, config, bias=module.bias is not None)
        layer.weight.data.copy_(module.weight.data.to(torch.bfloat16))
        if module.bias is not None:
            assert layer.bias is not None
            layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora_b(self.lora_a(x))
    
def apply_lora(model: nn.Module, config: Optional[LoraConfig] = None, device: str = "cpu"):
    if config is None:
        config = LoraConfig() 
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model, 
                name, 
                LoraBFloat16Linear(
                    module.in_features,
                    module.out_features,
                    config,
                    bias=module.bias is not None,
                ),
            )
        else:
            apply_lora(module, config)
    
    lora_model = model.to(device=device, dtype=torch.bfloat16)
    return lora_model






def apply_lora_peft(model: nn.Module, config: Optional[LoraConfig] = None, device: str = "cpu"):
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    if config is None:
        config = LoraConfig()
        
    for param in model.parameters():
        param.requires_grad_(False)

    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        inference_mode=True
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model