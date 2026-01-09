# imports
import os
from pathlib import Path
import cv2
import pandas as pd
import mediapipe as mp
import kagglehub
import numpy as np
from collections import defaultdict
from mp_utils import create_face_detector, crop_and_resize_face

# initializing params
FRAME_INTERVAL = 30          # sample 1 frame every 30
TARGET_SIZE = (224, 224)
OUT_DIR = "processed_daisee"
MAX_PER_CLASS = 1500         # cap per engagement class
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
splits = ["Train", "Validation", "Test"]
balanced = {}
rows = []


def get_daisee_path():
    path = kagglehub.dataset_download("olgaparfenova/daisee")
    return os.path.join(path, "DAiSEE") if os.path.exists(os.path.join(path, "DAiSEE")) else path


def load_labels(labels_dir):
    dfs = []
    for csv in Path(labels_dir).glob("*.csv"):
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        dfs.append(df[["ClipID", "Engagement"]])
    return pd.concat(dfs, ignore_index=True)


def preprocess():
    root = get_daisee_path()
    dataset_dir = os.path.join(root, "DataSet")
    labels_dir = os.path.join(root, "Labels")
    labels_df = load_labels(labels_dir)
    detector = create_face_detector()

    # -------------------------------
    # GLOBAL buffers (KEY CHANGE)
    # -------------------------------
    buffers = defaultdict(list)


    for split in splits:
        print(f"\nProcessing videos from: {split}")
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            continue

        videos = list(Path(split_path).rglob("*.avi"))

        for video in videos:
            row = labels_df[labels_df["ClipID"] == video.name]
            if row.empty:
                continue

            eng = int(row.iloc[0]["Engagement"])
            if eng not in {0, 1, 2, 3}:
                continue

            cap = cv2.VideoCapture(str(video))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_INTERVAL == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)
                    result = detector.detect(mp_img)

                    if result.detections:
                        d = max(result.detections, key=lambda x: x.categories[0].score)
                        bb = d.bounding_box
                        face = crop_and_resize_face(
                            frame,
                            int(bb.origin_x),
                            int(bb.origin_y),
                            int(bb.origin_x + bb.width),
                            int(bb.origin_y + bb.height),
                            TARGET_SIZE
                        )
                        buffers[eng].append(face)

                frame_idx += 1

            cap.release()

    detector.close()

    # -------------------------------
    # BALANCE PER CLASS (GLOBAL)
    # -------------------------------
    print("\nBalancing classes:")


    for label, faces in buffers.items():
        np.random.shuffle(faces)
        keep = min(len(faces), MAX_PER_CLASS)
        balanced[label] = faces[:keep]
        print(f"  class {label}: using {keep}")

    # -------------------------------
    # SPLIT INTO TRAIN / VAL / TEST
    # -------------------------------

    for label, faces in balanced.items():
        n = len(faces)
        n_train = int(TRAIN_RATIO * n)
        n_val = int(VAL_RATIO * n)

        split_faces = (
            [("Train", f) for f in faces[:n_train]] +
            [("Validation", f) for f in faces[n_train:n_train+n_val]] +
            [("Test", f) for f in faces[n_train+n_val:]]
        )

        for split, face in split_faces:
            out_dir = os.path.join(OUT_DIR, split, str(label))
            os.makedirs(out_dir, exist_ok=True)

            fname = f"{label}_{len(os.listdir(out_dir)):05d}.jpg"
            path = os.path.join(out_dir, fname)
            cv2.imwrite(path, face)

            rows.append({
                "image_path": path,
                "engagement": label,
                "split": split
            })

    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "dataset.csv"), index=False)

    print("\nFinal dataset:")
    print(df.groupby(["split", "engagement"]).size())


if __name__ == "__main__":
    preprocess()
