# imports
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import cv2
import time
from collections import deque

# Import MediaPipe utilities
from mp_utils import create_face_detector, detect_and_crop_face


class EngagementModel(nn.Module):
    """Same model architecture as in model1.py"""
    def __init__(self, num_classes=4):
        super(EngagementModel, self).__init__()
        # Use pretrained ResNet18 with weights parameter (not deprecated)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Load trained model
print("="*60)
print("Loading Trained Engagement Model...")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = EngagementModel(num_classes=4).to(device)

# Check if model exists
if not os.path.exists('best_engagement_model.pth'):
    print("\nERROR: Trained model not found!")
    print("Please run: python model1.py")
    print("This will train the model and save it as 'best_engagement_model.pth'")
    exit(1)

model.load_state_dict(torch.load('best_engagement_model.pth', map_location=device))
model.eval()
print("✓ Model loaded successfully!")

# Define engagement labels
engagement_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']

# Debug: Check model parameters to ensure it's not random
print("\nModel Debug Info:")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Test with a dummy input to see output distribution
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_output = model(dummy_input)
    dummy_probs = torch.softmax(dummy_output, dim=1)
    print(f"Dummy input output probabilities: {dummy_probs[0].cpu().numpy()}")
    print(f"Model biased towards class: {torch.argmax(dummy_probs).item()} ({engagement_labels[torch.argmax(dummy_probs).item()]})")

# Define transform (same as validation transform in model1.py)
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Webcam detection with trained model
print("\n" + "="*50)
print("Starting Live Webcam Engagement Detection...")
print("="*50)

PREDICTION_INTERVAL = 1  # Predict every 1 second
AVERAGE_INTERVAL = 10    # Calculate average every 10 seconds
CAMERA_INDEX = 0

my_detector = create_face_detector()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

next_prediction_time = time.time() + PREDICTION_INTERVAL
next_average_time = time.time() + AVERAGE_INTERVAL

# Store last 10 predictions for averaging
prediction_history = deque(maxlen=10)

print("Live webcam started. Press 'q' in the video window to quit.")
print(f"Predicting every {PREDICTION_INTERVAL}s, averaging every {AVERAGE_INTERVAL}s")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Live Webcam - Engagement Detection", frame)

    now = time.time()

    # Make prediction every second
    if now >= next_prediction_time:
        cropped_face, found = detect_and_crop_face(my_detector, frame, target_size=(224, 224))

        if found:
            # Predict engagement level
            face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            face_tensor = val_test_transform(face_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                confidence = probabilities[0][predicted.item()].item() * 100
                engagement = engagement_labels[predicted.item()]

            # Store prediction
            prediction_history.append({
                'class': predicted.item(),
                'label': engagement,
                'confidence': confidence,
                'timestamp': now
            })

            # Print current prediction
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {engagement} ({confidence:.1f}%)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] No face detected ❌")

        next_prediction_time = now + PREDICTION_INTERVAL

    # Calculate and print average every 10 seconds
    if now >= next_average_time and len(prediction_history) > 0:
        print("\n" + "="*60)
        print(f"10-SECOND AVERAGE REPORT [{time.strftime('%H:%M:%S')}]")
        print("="*60)

        # Calculate average engagement level
        class_counts = [0, 0, 0, 0]  # Count for each class
        for pred in prediction_history:
            class_counts[pred['class']] += 1

        # Calculate weighted average
        total_predictions = len(prediction_history)
        avg_class = sum(i * count for i, count in enumerate(class_counts)) / total_predictions
        avg_confidence = sum(pred['confidence'] for pred in prediction_history) / total_predictions

        # Find most frequent class
        most_frequent_class = class_counts.index(max(class_counts))
        most_frequent_label = engagement_labels[most_frequent_class]

        print(f"Total predictions in last 10s: {total_predictions}")
        print(f"\nClass Distribution:")
        for i, label in enumerate(engagement_labels):
            percentage = (class_counts[i] / total_predictions) * 100
            bar = "█" * int(percentage / 5)  # Visual bar
            print(f"  {label:15s}: {class_counts[i]:2d} ({percentage:5.1f}%) {bar}")

        print(f"\nAverage Engagement Score: {avg_class:.2f}")
        print(f"Most Frequent: {most_frequent_label} ({class_counts[most_frequent_class]} times)")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        print("="*60 + "\n")

        next_average_time = now + AVERAGE_INTERVAL

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
my_detector.close()

print("\n" + "="*60)
print("Application closed")
print("="*60)
