from pathlib import Path 
import torch 
from torch import nn

MASOANET_DIM = 2048

class LayerNorm(torch.nn.Module):

    num_channels: int
    eps: float

    def __init__(self, num_channels: int, eps: float = 1e-5, normalize: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if normalize:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))

            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)
        return rank
    
class CustomMasoaNet(torch.nn.Module):
    class MasoaBlock(torch.nn.Module):
        def __init__(self, channels, expansion=4, dropout=0.1):
            super().__init__()
            hidden = channels * expansion 
            self.model = nn.Sequential(
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, channels)
            )
        def forward(self, x):
            return self.model(x) + x
        
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            self.MasoaBlock(MASOANET_DIM),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM)
        )
        
    def forward(self, x):
        return self.model(x)
    
def load_network(path: Path):
    net = CustomMasoaNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net