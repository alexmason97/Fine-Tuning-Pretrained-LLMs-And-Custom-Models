from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys 
import torch

from . import MasoaNet, bf16MasoaNet, loraMasoaNet, quantMasoaNet, qloraMasoaNet

ALL_MODELS = {
    "MasoaNet": MasoaNet,
    "bf16MasoaNet": bf16MasoaNet,
    "loraMasoaNet": loraMasoaNet,
    "quantMasoaNet": quantMasoaNet,
    "qloraMasoaNet": qloraMasoaNet,
}


def load_model(model_name: str, path: Path | None) -> torch.nn.Module:
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model {model_name}")
    return ALL_MODELS[model_name].load_network(path)


@dataclass
class MemoryProfile:
    total: int = 0

    def __int__(self) -> int:
        return self.total

    def __str__(self) -> str:
        return f"{self.total / 1024.0 / 1024.0} MB"


@contextmanager
def memory_profile(device):
    if device == "cuda":
        mem = MemoryProfile()
        _mem_init = torch.cuda.memory_allocated()
        yield mem
        _mem_end = torch.cuda.memory_allocated()
        mem.total = _mem_end - _mem_init
    elif device == "cpu":
        from torch.profiler import profile
        mem = MemoryProfile()
        with profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            yield mem
        mem.total = prof.events().total_average().self_cpu_memory_usage
    else:
        raise ValueError(f"Unknown device {device}")


def num_parameters(model: torch.nn.Module) -> int:
    from itertools import chain
    # Number of parameters and buffers in a model.
    return sum(p.numel() for p in chain(model.buffers(), model.parameters()))


def mem_parameters(model: torch.nn.Module) -> int:
    from itertools import chain
    # Memory used for parameters and buffers in a model in bytes.
    return sum(p.numel() * p.element_size() for p in chain(model.buffers(), model.parameters()))


@dataclass
class ModelStats:
    num_parameters: int
    trainable_parameters: int
    theoretical_memory: float
    actual_memory: float
    forward_memory: float
    backward_memory: float

    @classmethod
    def from_model(cls, m: torch.nn.Module):
        original_device = next(m.parameters()).device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        m.to("cpu")
        with memory_profile(device) as mem_model:
            if device == "cpu":
                m_copy = deepcopy(m)
            else:
                m_copy = m.to(device)
        del m_copy

        x = torch.randn(2048, MasoaNet.MASOANET_DIM).to(device)

        with memory_profile(device) as mem_forward:
            with torch.no_grad():
                m(x)

        with memory_profile(device) as mem_backward:
            m(x).mean().backward()

        m.to(original_device)
        return cls(
            num_parameters=num_parameters(m),
            trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            theoretical_memory=mem_parameters(m) / 2**20,
            actual_memory=int(mem_model) / 2**20,
            forward_memory=int(mem_forward) / 2**20,
            backward_memory=int(mem_backward) / 2**20,
        )
        
def model_info(model_name1: str, *model_name2: str):
    model_names = [model_name1, *model_name2]
    stats = {}
    for model_name in model_names:
        model = load_model(model_name, None)
        stats[model_name] = ModelStats.from_model(model)
    print("                    ", " ".join([f"{model_name:^14s}" for model_name in stats.keys()]))
    print("Trainable params    ", " ".join([f"  {m.trainable_parameters / 1000000:8.2f} M  " for m in stats.values()]))
    print(
        "Non-trainable params",
        " ".join([f"  {(m.num_parameters - m.trainable_parameters) / 1000000:8.2f} M  " for m in stats.values()]),
    )
    print("Total params        ", " ".join([f"  {m.num_parameters / 1000000:8.2f} M  " for m in stats.values()]))
    print("Forward memory      ", " ".join([f"  {m.forward_memory:8.2f} MB " for m in stats.values()]))
    print("Backward memory     ", " ".join([f"  {m.backward_memory:8.2f} MB " for m in stats.values()]))
    print("Model disk size     ", " ".join([f"  {m.actual_memory:8.2f} MB " for m in stats.values()]))

if __name__ == "__main__":
    model_info(*sys.argv[1:])