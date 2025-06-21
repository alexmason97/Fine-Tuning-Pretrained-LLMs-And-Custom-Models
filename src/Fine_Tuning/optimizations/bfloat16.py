import torch
import torch.nn.functional as F 
from torch import nn 

class BFloat16Linear(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.weight.data = self.weight.data.to(torch.bfloat16)
        self.weight.requires_grad_(False)
        if bias:
            assert self.bias is not None 
            self.bias.data = self.bias.data.to(torch.bfloat16)
            self.bias.requires_grad_(False)
        
    @classmethod 
    def from_float(cls, module: nn.Linear):
        """Wanted to keep as BFloat16Linear subclass with inheritence 

        Args:
            module (nn.Linear): _description_
        """
        layer = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        layer.weight.data.copy_(module.weight.data.to(torch.bfloat16))
        if module.bias is not None:
            layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))
        return layer 
    
    def forward(self, x: torch.Tensor):
        x_bf16 = x.to(torch.bfloat16)
        forward_pass_res = F.linear(x_bf16, self.weight, self.bias)
        return forward_pass_res.to(x.dtype)
    
    
def apply_bfloat16(model: nn.Module):
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BFloat16Linear.from_float(module))
        else:
            apply_bfloat16(module)
    return model
        