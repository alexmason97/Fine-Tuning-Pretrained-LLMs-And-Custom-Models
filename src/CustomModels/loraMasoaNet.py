from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from Fine_Tuning.optimizations.lora import LoraBFloat16Linear
from .MasoaNet import MASOANET_DIM, LayerNorm

class LoRAMasoaNet(torch.nn.Module):
    class MasoaBlock(torch.nn.Module):
        def __init__(self, channels, lora_dim: int, expansion=4, dropout=0.1):
            super().__init__()
            hidden = channels * expansion
            self.model = nn.Sequential(
                LoraBFloat16Linear(channels, hidden, lora_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                LoraBFloat16Linear(hidden, channels, lora_dim)
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            self.MasoaBlock(MASOANET_DIM, lora_dim),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim),
            LayerNorm(MASOANET_DIM),
            self.MasoaBlock(MASOANET_DIM, lora_dim)
        )
        
    def forward(self, x):
        return self.model(x)
        
def load_network(path: Path):
    net = LoRAMasoaNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
