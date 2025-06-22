from pathlib import Path 
import tempfile 
from typing import Optional

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, logging 
logging.set_verbosity_error()

def download_hf_model(model_name: str, device: str = None, cache_dir: Optional[Path] = None,
                      quantize: bool = False):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) 
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True, load_in_4bit=quantize)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=True, load_in_4bit=quantize)
    # Was running into an issue trying to load file so needed a default method to download again
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, load_in_4bit=quantize)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, load_in_4bit=quantize)
    
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

def generate(model, tokenizer, prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 2,):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
    prompt_len = inputs["input_ids"].shape[-1]
    generated_text = outputs[0][prompt_len+1:]
    return tokenizer.decode(generated_text, skip_special_tokens=True)