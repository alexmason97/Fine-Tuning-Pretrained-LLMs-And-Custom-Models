import sys
import types
import pytest
from torch import nn

from Fine_Tuning.optimizations.qlora import (
    QLoraConfig,
    QLoRALinear,
    apply_qlora,
)

def test_qlora_config_defaults():
    cfg = QLoraConfig()
    assert cfg.rank == 32
    assert cfg.alpha == 16
    assert pytest.approx(0.05, rel=1e-6) == cfg.dropout


def test_qloralinear_grad_flags():
    cfg = QLoraConfig(rank=4, alpha=8, dropout=0.1)
    layer = QLoRALinear(in_features=5, out_features=7, lora_dim=cfg.rank)
    assert layer.lora_a.weight.requires_grad
    assert layer.lora_b.weight.requires_grad
    if layer.bias is not None:
        assert not layer.bias.requires_grad


def test_apply_qlora_replaces_linear_layers(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

    replaced = []

    def fake_from_float(cls, module, cfg):
        replaced.append(module)
        return nn.Identity()

    monkeypatch.setattr(QLoRALinear, "from_float", classmethod(fake_from_float))
    q_model = apply_qlora(model, QLoraConfig(rank=2, alpha=2, dropout=0.0))

    assert all(isinstance(m, nn.Linear) for m in replaced)
    assert all(isinstance(child, nn.Identity) for child in q_model.children())
