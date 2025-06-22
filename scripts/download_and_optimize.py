from argparse import ArgumentParser 

import torch 
import warnings
warnings.filterwarnings("ignore")

from Fine_Tuning.model_utils import download_hf_model, generate, model_file_size
from Fine_Tuning.optimizations.bfloat16 import apply_bfloat16, apply_bfloat16_torch, BFloat16Linear
from Fine_Tuning.optimizations.quantization import quantize_4bit


# MODEL_NAME = "facebook/opt-350m"
MODEL_NAME = "meta-llama/Llama-3.2-1B"


def main() -> None:
    parser = ArgumentParser(description="Task to Downlaod and optimize the opt-350m model")
    parser.add_argument("prompt", help="Prompt here ot run through the models")
    parser.add_argument(
        "--impl",
        choices=["scratch", "reference"],
        default="reference",
        help="Use custom implementations or reference libraries",
    )
    args = parser.parse_args()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = download_hf_model(MODEL_NAME, device=device)
    
    base_size = model_file_size(model)
    print("Base model output:") 
    print(f"Prompt: {args.prompt}")
    print(f"Response:")
    print(generate(model, tokenizer, args.prompt))
    print(f"Base model size: {base_size / 1e6:.2f} MB") # bytes to MB for better viz
    
    if args.impl == "scratch":
        bf16_model = apply_bfloat16(model)
        quantization_model = quantize_4bit(model)
    else:
        bf16_model = apply_bfloat16_torch(model, device)
        quantization_model, _ = download_hf_model(MODEL_NAME, device, quantize=True)
    print("\n")
    print("BFloat16 implementation model output:") 
    print(f"Prompt: {args.prompt}")
    print(f"Response:")
    print(generate(bf16_model, tokenizer, args.prompt)) 
    print(f"BFloat16 model size: {model_file_size(bf16_model) / 1e6:.2f} MB")
        
    print("\n")
    print("Quantized model output:")
    print(f"Prompt: {args.prompt}")
    print(f"Response:")
    print(generate(quantization_model, tokenizer, args.prompt))
    print(f"Quantized model size: {model_file_size(quantization_model) / 1e6:.2f} MB")
    
if __name__ == "__main__":
    main()