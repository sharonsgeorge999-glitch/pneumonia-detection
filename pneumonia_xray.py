"""
Pneumonia detection from chest X-rays (transfer learning with ResNet18, PyTorch).

Dataset: Kaggle "Chest X-Ray Images (Pneumonia)" by Paul Mooney
Expected layout:
    chest_xray/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL,   val/PNEUMONIA
        test/NORMAL,  test/PNEUMONIA

Install:  pip install torch torchvision scikit-learn pillow
Train:    python pneumonia_xray.py --epochs 8
          (data path defaults to C:\AI training\chest_xray; override with --data)
Predict:  python pneumonia_xray.py --predict path/to/xray.jpg
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "pneumonia_resnet18.pt"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # NORMAL vs PNEUMONIA
    return model.to(DEVICE)


def train(data_dir, epochs, batch_size=32, lr=1e-4):
    # No 'val' folder needed: we hold out 15% of the train folder for validation
    full_train = datasets.ImageFolder(f"{data_dir}/train", train_tf)  # with augmentation
    full_val = datasets.ImageFolder(f"{data_dir}/train", eval_tf)     # same images, no augmentation
    test_ds = datasets.ImageFolder(f"{data_dir}/test", eval_tf)
    print("Classes:", full_train.class_to_idx)

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(0.15 * len(perm))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)
    train_ds.classes = full_train.classes  # used later for saving / reports

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size, num_workers=0)

    # Dataset is imbalanced (~3x more pneumonia) -> class weights
    counts = torch.bincount(torch.tensor([full_train.targets[i] for i in train_idx]))
    weights = (counts.sum() / (2.0 * counts)).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        val_acc = evaluate(model, val_dl)[0]
        print(f"Epoch {epoch+1}/{epochs}  loss={running/len(train_ds):.4f}  val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, MODEL_PATH)

    # Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model"])
    acc, y_true, y_pred = evaluate(model, test_dl)
    print(f"\nTest accuracy: {acc:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=train_ds.classes))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        preds = model(x.to(DEVICE)).argmax(1).cpu()
        y_true += y.tolist()
        y_pred += preds.tolist()
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return acc, y_true, y_pred


@torch.no_grad()
def predict(image_path):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval()
    img = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(img), dim=1)[0]
    idx = probs.argmax().item()
    print(f"Prediction: {ckpt['classes'][idx]}  (confidence {probs[idx]:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\AI training\chest_xray\chest_xray")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--predict", help="path to a single X-ray image")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        train(args.data, args.epochs)
