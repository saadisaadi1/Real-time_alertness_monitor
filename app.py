# imports
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import cv2
import time
from collections import deque
from mp_utils import create_face_detector, detect_and_crop_face


class EngagementModel(nn.Module):
    """Same model architecture as in Resnet18.py"""
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone  # Changed from self.backbone to self.model

    def forward(self, x):
        return self.model(x)  # Changed from self.backbone to self.model


def main():
    # ============================
    # Load trained model
    # ============================
    print("=" * 60)
    print("Loading Trained Engagement Model...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EngagementModel(num_classes=4).to(device)

    if not os.path.exists("best_engagement_model.pth"):
        print("\nERROR: Trained model not found!")
        print("Please run: python Resnet18.py")
        return

    model.load_state_dict(
        torch.load("best_engagement_model.pth", map_location=device)
    )
    model.eval()
    print("Model loaded successfully!")

    engagement_labels = ["Low", "Medium-Low", "Medium-High", "High"]

    print("\nModel Debug Info:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        probs = torch.softmax(model(dummy_input), dim=1)
        print("Dummy probs:", probs[0].cpu().numpy())

    # ============================
    # Transforms
    # ============================
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ============================
    # Webcam setup
    # ============================
    print("\n" + "=" * 50)
    print("Starting Live Webcam Engagement Detection...")
    print("=" * 50)

    PREDICTION_INTERVAL = 1
    AVERAGE_INTERVAL = 10
    CAMERA_INDEX = 0

    my_detector = create_face_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        my_detector.close()
        return

    next_prediction_time = time.time() + PREDICTION_INTERVAL
    next_average_time = time.time() + AVERAGE_INTERVAL

    prediction_history = deque(maxlen=10)

    print("Live webcam started. Press 'q' to quit.")

    # ============================
    # Main loop
    # ============================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow("Live Webcam - Engagement Detection", frame)
        now = time.time()

        if now >= next_prediction_time:
            cropped_face, found = detect_and_crop_face(
                my_detector, frame, target_size=(224, 224)
            )

            if found:
                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                face_tensor = val_test_transform(face_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()

                confidence = probs[0, pred].item() * 100
                label = engagement_labels[pred]

                prediction_history.append({
                    "class": pred,
                    "confidence": confidence
                })

                print(f"[{time.strftime('%H:%M:%S')}] {label} ({confidence:.1f}%)")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No face detected")

            next_prediction_time = now + PREDICTION_INTERVAL

        if now >= next_average_time and prediction_history:
            counts = [0, 0, 0, 0]
            for p in prediction_history:
                counts[p["class"]] += 1

            total = sum(counts)
            avg_class = sum(i * c for i, c in enumerate(counts)) / total

            print("\n" + "=" * 60)
            print("10-SECOND AVERAGE")
            print("Counts:", counts)
            print(f"Average engagement score: {avg_class:.2f}")
            print("=" * 60 + "\n")

            next_average_time = now + AVERAGE_INTERVAL

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ============================
    # Cleanup
    # ============================
    cap.release()
    cv2.destroyAllWindows()
    my_detector.close()

    print("\n" + "=" * 60)
    print("Application closed")
    print("=" * 60)


if __name__ == "__main__":
    main()
