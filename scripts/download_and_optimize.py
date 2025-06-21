from argparse import ArgumentParser 

import torch 

from Fine_Tuning.model_utils import (
    LoraConfig,
    QLoraConfig,
    apply_bfloat16,
    apply_lora,
    apply_qlora,
    download_hf_model,
    generate,
    model_file_size,
    quantize_4bit,
    quantize_dynamic,
)

MODEL_NAME = "Llama-3.2-1B"

def main() -> None:
    parser = ArgumentParser(description="Task to Downlaod and optimize the Llama-3.2-1B model")
    parser.add_argument("prompt", help="Prompt here ot run through the models")
    parser.add_argument(
        "--quant-method",
        choices=["dynamic", "manual"],
        default="dynamic",
        help="Quantization method: dynamic or manual 4-bit",
    )
    args = parser.parse_args()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = download_hf_model(MODEL_NAME, device)
    
    base_size = model_file_size(model) 
    print(f"Base model size: {base_size / 1e6:.2f} MB") # bytes to MB 
    print("Base model output:") 
    print(generate(model, tokenizer, args.prompt))
    
    bf16_model = apply_bfloat16(model) 
    print("\n")
    print("BFloat16 implementation model output:") 
    print(generate(bf16_model, tokenizer, args.prompt)) 
    print(f"BFloat16 model size: {model_file_size(bf16_model) / 1e6:.2f} MB")
    
    lora_model = apply_lora(model, LoraConfig())
    print("\n")
    print("LoRA implementation model output:")
    print(generate(lora_model, tokenizer, args.prompt)) 
    print(f"LoRA model size: {model_file_size(lora_model) / 1e6:.2f} MB")
        
    if args.quant_method == "dynamic":
        quantization_model = quantize_dynamic(model)
    else:
        quantization_model = quantize_4bit(model)   
    print("\n")
    print("Quantized model output:")
    print(generate(lora_model, tokenizer, args.prompt)) 
    print(f"Quantized model size: {model_file_size(lora_model) / 1e6:.2f} MB")
    
    try:
        qlora_model = apply_qlora(model, QLoraConfig())
        print("\n")
        print("QLoRA implementation model output:") 
        print(generate(bf16_model, tokenizer, args.prompt)) 
        print(f"BFloat16 model size: {model_file_size(bf16_model) / 1e6:.2f} MB")
    except ImportError as e:
        print(f"QLoRA not available: {e}")
    
    
if __name__ == "__main__":
    main()