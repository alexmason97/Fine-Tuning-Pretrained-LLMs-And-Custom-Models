import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms, models

from CustomModels.MasoaNet import CustomMasoaNet , MASOANET_DIM
from CustomModels.loraMasoaNet import LoRAMasoaNet
from CustomModels.qloraMasoaNet import QLoRAMasoaNet


MODEL_CLASSES = {
     "MasoaNet": CustomMasoaNet,
     "loraMasoaNet": LoRAMasoaNet,
     "qloraMasoaNet": QLoRAMasoaNet
 }

def load_random_samples(n_samples, batch_size, device):
    x = torch.randn(n_samples, MASOANET_DIM, device=device)
    y = torch.cat([
        torch.zeros(n_samples // 2, dtype=torch.float32, device=device),
        torch.ones(n_samples - n_samples // 2, dtype=torch.float32, device=device)
    ])
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), None


def make_MASOANET_loader(batch_size, device):
    # cifar-10 label 3=cat, 5=dog
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    cifar = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    idx = [i for i, (_, lbl) in enumerate(cifar) if lbl in (3, 5)]
    ds = Subset(cifar, idx)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # We just need the backbone for feature extraction to help our MasoaNet head get the right features.
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return loader, backbone


def fit_binary_classifier(
    head_model: nn.Module,
    loader: DataLoader,
    backbone: nn.Module = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_model = head_model.to(device)

    # frozen linear classifier on top of the 2048-dim feat
    classifier = nn.Linear(MASOANET_DIM, 1, bias=False).to(device)
    classifier.requires_grad_(False)

    opt = optim.AdamW(head_model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        head_model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            if backbone is None:
                feats, labels = batch
                feats  = feats.to(device)
                labels = labels.to(device).float()
            else:
                imgs, labs = batch
                imgs = imgs.to(device)
                with torch.no_grad():
                    feats = backbone(imgs)    
                labels = (labs.to(device) == 5).float()

            opt.zero_grad()
            out = head_model(feats)
            logits = classifier(out).squeeze(-1)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * feats.size(0)
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += feats.size(0)

        avg_loss = total_loss/total
        acc = correct/total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    return best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=["random","cifar"], default="random")
    p.add_argument("--model",      choices=list(MODEL_CLASSES), default="baseline",
                     help="Which model variant to train")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save",       type=str, default="masoanet_ckpt.pth")
    p.add_argument("--save-dir",   type=str, default="checkpoints",
                   help="Where to write the final .pth")
    p.add_argument("--n_samples",  type=int,   default=1000)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     type=str,   default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else None
    
    model_cls = MODEL_CLASSES[args.model] 
    model = model_cls()

    if args.dataset == "random":
        loader, backbone = load_random_samples(
            args.n_samples, args.batch_size,
            device or torch.device("cpu")
        )
    else:
        loader, backbone = make_MASOANET_loader(
            args.batch_size,
            device or torch.device("cpu")
        )

    if args.checkpoint and Path(args.checkpoint).exists():
        print("Loading head weights from", args.checkpoint)
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device),
            strict=True
        )

    best = fit_binary_classifier(
        model, loader,
        backbone=backbone,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    print(f"\nBest acc = {best:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{args.model}.pth"
    torch.save(model.state_dict(), str(out))
    print("Saved head to", out)


if __name__=="__main__":
    main()
