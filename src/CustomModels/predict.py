import argparse
import random 
import torch
from torch import nn 
from pathlib import Path 
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage
from torch.utils.data import Subset
from CustomModels.MasoaNet import CustomMasoaNet, MASOANET_DIM, load_network as load_masoa
from CustomModels.loraMasoaNet import LoRAMasoaNet, load_network as load_lora
from CustomModels.qloraMasoaNet import QLoRAMasoaNet, load_network as load_qlora
from CustomModels.quantMasoaNet import QuantMasoaNet, load_network as load_quant
from CustomModels.bf16MasoaNet  import BF16MasoaNet, load_network as load_bf16

MODEL_CLASSES = {
    "MasoaNet": (CustomMasoaNet,load_masoa),
    "loraMasoaNet": (LoRAMasoaNet,load_lora),
    "qloraMasoaNet": (QLoRAMasoaNet,load_qlora),
    "quantMasoaNet": (QuantMasoaNet,load_quant),
    "bf16MasoaNet": (BF16MasoaNet,load_bf16),
}

def main():
    parser = argparse.ArgumentParser(
        description= "Running inference on Masoa Models on a random CIFAR10 cat/dog image"
    )
    parser.add_argument("--model", choices=list(MODEL_CLASSES), required=True,
                        help="Model we want to load for inference")
    parser.add_argument("--file_name", type=Path, required=True, 
                        help="Default path to the .pth pytorch model file for running inference")
    parser.add_argument("--device", type=str, default=None,
                        help="device to run inference on. e.g. 'cude' or 'cpu'; defaults to CUDA if available")
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    _, loader_fn = MODEL_CLASSES[args.model]
    model = loader_fn(args.file_name)
    model.to(device)
    if args.model == "bf16MasoaNet":
        # bfloat16 cast
        model = model.to(torch.bfloat16)
    model.eval()
    
    # utilize frozen ResNet-50 Backbone from original model training 
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False 
        
    # dataset normalization for subset of cat/dog images 
    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar_test = datasets.CIFAR10("data", train=False, download=True, transform=tfs)
    index = [i for i, (_, label) in enumerate(cifar_test) if label in (3, 5)]
    subset = Subset(cifar_test, index)
    
    # Random image selection 
    img_tensor, true_label = subset[random.randrange(len(subset))]
    
    to_pil = ToPILImage()
    # Looked up how to denormalize our images properly for the PIL format
    denorm = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5,  1/0.5,  1/0.5]
    )
    img_denorm = denorm(img_tensor)
    pil_img = to_pil(img_denorm)
    pil_img.show()
    
    with torch.no_grad():
        feats = backbone(img_tensor.unsqueeze(0).to(device)) # Outputvec shape of size (1, 2048)
        if args.model == "bf16MasoaNet":
            feats = feats.to(torch.bfloat16)
        output = model(feats)
        classifier = nn.Linear(MASOANET_DIM, 1, bias=False).to(device)
        if args.model == "bf16MasoaNet":
            classifier = classifier.to(torch.bfloat16)
        classifier.eval()
        
        class_logits = classifier(output).squeeze(1) 
        class_prob = torch.sigmoid(class_logits).item()         
        
    pred_label = 5 if class_prob > 0.5 else 3 
    label_dict = {3: "cat", 5: "dog"}
    
    print(f"True class: {label_dict[true_label]} ({true_label})")
    print(f"Predicted class: {label_dict[pred_label]} ({pred_label})")
    print(f"Probability of '{label_dict[true_label]}': {class_prob:.4f}")
    
if __name__ == "__main__":
    main() 