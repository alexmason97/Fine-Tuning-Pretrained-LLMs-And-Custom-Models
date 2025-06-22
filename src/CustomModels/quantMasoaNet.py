from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from Fine_Tuning.optimizations.quantization import Linear4Bit
from .MasoaNet import MASOANET_DIM, LayerNorm

class QuantMasoaNet(torch.nn.Module):
    class MasoaBlock(torch.nn.Module):
        def __init__(self, channels, expansion=4, dropout=0.1):
            super().__init__()
            hidden = channels * expansion
            self.model = nn.Sequential(
                Linear4Bit(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                Linear4Bit(hidden, channels)
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
    net = QuantMasoaNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net