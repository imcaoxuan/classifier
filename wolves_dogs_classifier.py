import os
import zipfile
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from albumentations import (
    Compose, Resize, HorizontalFlip, RandomBrightnessContrast,
    GaussNoise, Normalize
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm import tqdm


def set_up_data():
    with zipfile.ZipFile('dataset/wo.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_path = Path(root)
        self.transform = transform
        self.samples = []

        for label, cls in enumerate(['dogs', 'wolves']):
            folder = self.root_path.joinpath(cls)
            for fname in os.listdir(folder):
                self.samples.append((os.path.join(folder, fname), label))
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)


def train():
    train_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(),
        GaussNoise(p=0.2),
        Normalize(),
        ToTensorV2(),
    ])

    val_transform = Compose([
        Resize(224, 224),
        Normalize(),
        ToTensorV2(),
    ])

    train_ds = ImageDataset('dataset/wolves_and_dogs/train', transform=train_transform)
    val_ds = ImageDataset('dataset/wolves_and_dogs/val', transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train_one_epoch():
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate():
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                _, predicted = torch.max(preds, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    best_acc = 0

    for epoch in range(10):
        loss = train_one_epoch()
        acc = validate()

        print(f'Epoch {epoch + 1}, Loss={loss:.4f}, Val Acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'weights/best_model_2s.pth')
            print('✓ Saved new best model')

    torch.save(model, 'weights/last_model_2s.pt')


if __name__ == '__main__':
    # set_up_data()
    train()
