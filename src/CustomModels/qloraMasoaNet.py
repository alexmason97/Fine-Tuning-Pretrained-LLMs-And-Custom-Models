from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from Fine_Tuning.optimizations.qlora import QLoRALinear
from .MasoaNet import MASOANET_DIM, LayerNorm

class QLoRAMasoaNet(torch.nn.Module):
    class MasoaBlock(torch.nn.Module):
        def __init__(self, channels, lora_dim: int, group_size, expansion=4, dropout=0.1):
            super().__init__()
            hidden = channels * expansion
            self.model = nn.Sequential(
                QLoRALinear(channels, hidden, lora_dim, group_size),
                nn.GELU(),
                nn.Dropout(dropout),
                QLoRALinear(hidden, channels, lora_dim, group_size)
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = nn.Sequential(
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim, group_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
def load_network(path: Path):
    net = QLoRAMasoaNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net