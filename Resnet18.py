# ============================================================
# IMPORTS
# ============================================================

import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ============================================================
# HYPER-PARAMETERS
# ============================================================

DATASET_DIR = "processed_daisee"
CSV_PATH = f"{DATASET_DIR}/dataset.csv"

NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 16              # Reduced for better generalization
NUM_EPOCHS = 25              # Increased for better convergence
LEARNING_RATE = 5e-4         # Lower learning rate for stability
WEIGHT_DECAY = 1e-4          # Added regularization
LR_STEP_SIZE = 8             # Adjust LR less frequently
LR_GAMMA = 0.5               # More gradual LR decay
NUM_WORKERS = 4
PIN_MEMORY = True

# ============================================================
# DATASET
# ============================================================

class DAiSEEDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = int(row["engagement"])
        return image, label


# ============================================================
# MODEL
# ============================================================

class EngagementModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        # Add dropout for regularization
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone.fc.in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # Load dataset
    # -------------------------------
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "Train"]
    val_df   = df[df["split"] == "Validation"]
    test_df  = df[df["split"] == "Test"]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # -------------------------------
    # Transforms
    # -------------------------------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2)
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------------
    # Dataloaders
    # -------------------------------
    train_loader = DataLoader(
        DAiSEEDataset(train_df, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda"
    )

    val_loader = DataLoader(
        DAiSEEDataset(val_df, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        DAiSEEDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_weights = torch.tensor([1500/600, 1.0, 1.0, 1.0])

    # -------------------------------
    # Model / Optimizer
    # -------------------------------
    model = EngagementModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA
    )

    # -------------------------------
    # Training
    # -------------------------------
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device
        )

        scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_engagement_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -------------------------------
    # Test
    # -------------------------------
    model.load_state_dict(torch.load("best_engagement_model.pth"))
    _, test_acc = run_epoch(
        model, test_loader, criterion, None, device
    )

    print(f"\nTest Accuracy: {test_acc:.2f}%")
