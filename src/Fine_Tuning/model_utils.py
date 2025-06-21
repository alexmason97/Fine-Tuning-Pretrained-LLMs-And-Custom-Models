from pathlib import Path 
import tempfile 
from typing import Optional

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

from .optimizations.lora import LoraConfig, apply_lora 
from .optimizations.lora import QLoraConfig, apply_qlora 
from .optimizations.quantization import (
    Linear4Bit, 
    block_dequantize_4bit, 
    block_quantize_4bit, 
    quantize_4bit, 
    quantize_dynamic,
)

def download_hf_model(model_name: str, device: str = None, cache_dir: Optional[Path] = None,):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) 
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True)
    # Was running into an issue trying to load file so needed a default method to download again
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir)
    
    model.to(device)
    return model, tokenizer 

def model_file_size(model: torch.nn.Module):
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        # Grabs all model params
        torch.save(model.state_dict(), tmp.name) 
        # grab file size 
        size = Path(tmp.name).stat().st_size 
    # Remove temp file from disk to give me some space back on my SSD
    Path(tmp.name).unlink(missing_ok=True) 
    return size 

def generate(model, tokenizer, prompt: str, output_token_length_max: int = 200):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=output_token_length_max)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)