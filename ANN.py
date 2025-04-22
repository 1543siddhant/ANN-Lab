# Imports
import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as ET

# 1. Custom Dataset for VOC‑style annotations
class CatDogDataset(Dataset):
    def __init__(self, root="Data", transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # Parse corresponding XML
        xml_path = os.path.join(self.root, "annotations",
                                self.imgs[idx].replace(".jpg", ".xml"))
        tree = ET.parse(xml_path)
        boxes, labels = [], []
        for obj in tree.findall("object"):
            name = obj.find("name").text
            labels.append(1 if name == "dog" else 2)  # 1=dog, 2=cat
            bb = obj.find("bndbox")
            boxes.append([int(bb.find(x).text) for x in ("xmin","ymin","xmax","ymax")])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        # Apply transform
        img = self.transforms(img) if self.transforms else ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# 2. DataLoader
dataset = CatDogDataset(root="Data", transforms=ToTensor())
data_loader = DataLoader(dataset, batch_size=4, shuffle=True,
                         collate_fn=lambda x: tuple(zip(*x)))

# 3. Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)  # PyTorch Faster R‑CNN :contentReference[oaicite:2]{index=2}
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
model.to(device)

# 4. Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# 5. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")



# https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection

