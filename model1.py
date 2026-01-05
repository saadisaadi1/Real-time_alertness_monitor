import numpy as np
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import cv2
import time
from sklearn.model_selection import train_test_split

# Import preprocessing functions
from preprocessing import load_processed_dataset


# ============================================================
# CLASS DEFINITIONS
# ============================================================

class DAiSEEDataset(Dataset):
    """PyTorch Dataset for processed DAiSEE images"""

    def __init__(self, dataframe, transform=None, target_col='Engagement'):
        self.df = dataframe
        self.transform = transform
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['processed_path']

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label (default to 0 if not available)
        label = row.get(self.target_col, 0)

        return image, label


class EngagementModel(nn.Module):
    """Neural network model for engagement detection"""

    def __init__(self, num_classes=4):
        super(EngagementModel, self).__init__()
        # Use pretrained ResNet18 with weights parameter (not deprecated)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    # GPU Configuration for RTX 3050
    print("\n" + "="*50)
    print("GPU Configuration")
    print("="*50)

    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        device = torch.device('cuda')
    else:
        print("⚠ CUDA is not available. Using CPU.")
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print("="*50)

    # Load preprocessed dataset
    print("\n" + "="*50)
    print("Loading Preprocessed DAiSEE Dataset...")
    print("="*50)

    try:
        processed_df = load_processed_dataset("processed_daisee")
    except FileNotFoundError:
        print("\nPreprocessed dataset not found!")
        print("Please run: python preprocessing.py")
        print("This will process the dataset and save it for training.")
        sys.exit(1)

    # **NEW: Custom 80/10/10 split instead of using pre-defined splits**
    print("\n" + "="*50)
    print("Creating 80/10/10 Train/Val/Test Split...")
    print("="*50)

    # Shuffle and split
    processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # First split: 80% train, 20% temp (for val+test)
    train_df, temp_df = train_test_split(processed_df, test_size=0.2, random_state=42)

    # Second split: Split temp into 50% val, 50% test (each 10% of total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nDataset split complete:")
    print(f"  Total images: {len(processed_df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(processed_df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} ({len(val_df)/len(processed_df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(processed_df)*100:.1f}%)")

    # Check label distribution in each split
    print(f"\nEngagement label distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n  {split_name}:")
        for level in range(4):
            count = len(split_df[split_df['Engagement'] == level])
            print(f"    Level {level}: {count} samples ({count/len(split_df)*100:.1f}%)")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = DAiSEEDataset(train_df, transform=train_transform)
    val_dataset = DAiSEEDataset(val_df, transform=val_test_transform)
    test_dataset = DAiSEEDataset(test_df, transform=val_test_transform)

    # Create dataloaders with pin_memory for faster GPU transfer
    batch_size = 32
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Build model
    print("\n" + "="*50)
    print("Building Model...")
    print("="*50)

    model = EngagementModel(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)

    num_epochs = 10
    best_val_acc = 0.0
    train_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        # Print GPU memory usage if using CUDA
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_engagement_model.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

            # Debug: Check if labels are balanced in validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

            print(f"  Validation label distribution:")
            for i, label in enumerate(['Low', 'Medium-Low', 'Medium-High', 'High']):
                true_count = sum(1 for l in all_labels if l == i)
                pred_count = sum(1 for p in all_preds if p == i)
                print(f"    {label}: True={true_count}, Predicted={pred_count}")

        # Clear GPU cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_time = time.time() - train_start
    print("\n" + "="*60)
    print(f"Training Complete! Total time: {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)

    # Test the model
    print("\n" + "="*50)
    print("Testing Model...")
    print("="*50)

    model.load_state_dict(torch.load('best_engagement_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Check class distribution in test set
    print("\nTest Set Analysis:")
    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.numpy())

    print(f"Test label distribution:")
    for i, label in enumerate(['Low', 'Medium-Low', 'Medium-High', 'High']):
        true_count = sum(1 for l in all_test_labels if l == i)
        pred_count = sum(1 for p in all_test_preds if p == i)
        accuracy = sum(1 for l, p in zip(all_test_labels, all_test_preds) if l == i and p == i)
        if true_count > 0:
            class_acc = 100. * accuracy / true_count
        else:
            class_acc = 0
        print(f"  {label}: True={true_count}, Predicted={pred_count}, Class Acc={class_acc:.1f}%")

    print("\n" + "="*60)
    print("Model training complete!")
    print("Saved model: best_engagement_model.pth")
    print("You can now use this model in app.py for live video prediction")
    print("="*60)
