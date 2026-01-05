import numpy as np
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_PATH = "blaze_face_short_range.tflite"


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    """Download MediaPipe face detection model if not already present."""
    if os.path.exists(path):
        return path
    print("Downloading MediaPipe face model...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    print("Saved model to:", path)
    return path


def create_face_detector():
    """Create and return a MediaPipe face detector instance."""
    model_path = ensure_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    return vision.FaceDetector.create_from_options(options)


def crop_and_resize_face(frame_bgr, x1, y1, x2, y2, target_size=(224, 224)):
    """
    Crop a face region from frame and resize to target size for training.

    Args:
        frame_bgr: Input image in BGR format
        x1, y1, x2, y2: Bounding box coordinates
        target_size: Tuple (width, height) for resizing, default (224, 224)

    Returns:
        Resized cropped face image
    """
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Crop the face region
    cropped_face = frame_bgr[y1:y2, x1:x2]

    # Resize to target size for training
    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_face


def detect_and_crop_face(detector, frame_bgr, target_size=(224, 224)):
    """
    Detect face in frame, crop and resize it.

    Args:
        detector: MediaPipe face detector instance
        frame_bgr: Input frame in BGR format
        target_size: Target size for resizing (width, height)

    Returns:
        Tuple of (cropped_resized_face, found_flag)
    """
    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect(mp_image)
    if not result.detections:
        return None, False

    # Take best detection by confidence
    best = max(result.detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box
    x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
    x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)

    # Crop and resize
    resized_face = crop_and_resize_face(frame_bgr, x1, y1, x2, y2, target_size=target_size)

    return resized_face, True

