#!/usr/bin/env python3
import argparse
import random
import torch
from torch import nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage
from torch.utils.data import Subset
from CustomModels.MasoaNet import CustomMasoaNet, MASOANET_DIM
from CustomModels.loraMasoaNet import LoRAMasoaNet
from CustomModels.qloraMasoaNet import QLoRAMasoaNet

MODEL_CLASSES = {
    "MasoaNet": CustomMasoaNet,
    "loraMasoaNet": LoRAMasoaNet,
    "qloraMasoaNet": QLoRAMasoaNet
}

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a random CIFAR10 cat/dog image"
    )
    parser.add_argument("--model", choices=list(MODEL_CLASSES), required=True,
                        help="Which head-model to load")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the .pth checkpoint of the head model")
    parser.add_argument("--device", type=str, default=None,
                        help="e.g., 'cuda' or 'cpu'; defaults to CUDA if available")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = MODEL_CLASSES[args.model]()                      
    state = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(state)                           
    head.to(device).eval()

    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    cifar_test = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    idx = [i for i, (_, lbl) in enumerate(cifar_test) if lbl in (3,5)]
    subset = Subset(cifar_test, idx)

    img_tensor, true_lbl = subset[random.randrange(len(subset))]
    to_pil = ToPILImage()
    denorm = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5,  1/0.5,  1/0.5]
    )
    img_denorm = denorm(img_tensor)
    pil_img = to_pil(img_denorm)
    pil_img.show()

    with torch.no_grad():
        feats = backbone(img_tensor.unsqueeze(0).to(device)) 
        out   = head(feats)                                  
        # here we assume your head outputs features that you then feed
        # to a binary classifier; if your head already ends in logits,
        # skip the classifier below.
        classifier = nn.Linear(MASOANET_DIM, 1, bias=False).to(device)
        # if you saved classifier weights, load them here:
        # classifier.load_state_dict(torch.load("classifier.pth") )
        classifier.eval()

        logits = classifier(out).squeeze(1)   # scalar logit
        prob   = torch.sigmoid(logits).item()

    pred_lbl = 5 if prob > 0.5 else 3
    label_map = {3: "cat", 5: "dog"}

    print(f"True class:      {label_map[true_lbl]} ({true_lbl})")
    print(f"Predicted class: {label_map[pred_lbl]} ({pred_lbl})")
    print(f"Probability of 'dog': {prob:.4f}")

if __name__ == "__main__":
    main()
