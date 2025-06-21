from dataclasses import dataclass
from typing import Optional 

import math 

import torch 
from torch import nn 

@dataclass
class LoraConfig:
    
    rank: int = 4
    alpha: int = 32
    dropout: float = 0.0 
    
class LoraLinear(nn.Linear):
    
    def __init__(
        self, in_features: int, out_features: int, config: LoraConfig, **kwargs: int
        ):
        super().__init__(in_features, out_features)
        self.rank = config.rank
        self.alpha = config.alpha 
        self.scaling = self.alpha / self.rank 
        self.lora_dropout = nn.Dropout(p=config.dropout) #Should avoid for now (don't want misrepresented issues w/o checkpoints)
        self.lora_A = nn.Parameter(torch.zeros((self.rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.rank)))
        # need to initialize dist with sqrt(5) to match PyTorch init w/ weights  
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor):
        forward_pass_result = super().forward(x)
        if self.rank > 0:
            lora_output = self.lora_B @ self.lora_A
            lora_output = self.lora_dropout(x) @ lora_output.T 
            forward_pass_result = forward_pass_result + lora_output * self.scaling
        return forward_pass_result
    
def apply_lora(model: nn.Module, config: Optional[LoraConfig] = None):
    
    if config is None:
        config = LoraConfig() 
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model, 
                name, 
                LoraLinear(
                    module.in_features,
                    module.out_features,
                    config,
                    bias=module.bias is not None,
                ),
            )
        else:
            apply_lora(module, config)
    return model
        
        