import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

IMG_WIDTH = 1319
IMG_HEIGHT = 339
TARGET_WIDTH = 128
TARGET_HEIGHT = 32
ANNOTATION_FILES = [("annotations_gs06.csv", "Dataset/GS06"),
                    ("annotations_gs54.csv", "Dataset/GS54")]

# === DATA LOADING ===


def load_annotations(annotation_path, image_dir, target_size):
    df = pd.read_csv(annotation_path, header=None, names=[
        "frameNumber", "personID", "bodyLeft", "bodyTop", "bodyWidth", "bodyHeight"
    ])

    X, y, paths = [], [], []
    used_frames = set()

    for _, row in df.iterrows():
        frame_number = int(row["frameNumber"])
        if (image_dir, frame_number) in used_frames:
            continue
        used_frames.add((image_dir, frame_number))

        filename = f"frame_{frame_number:04d}.jpg"
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        try:
            img = cv2.resize(img, target_size)
        except Exception:
            continue

        img = img.astype(np.float32) / 255.0

        x_min = row["bodyLeft"]
        y_min = row["bodyTop"]
        w = row["bodyWidth"]
        h = row["bodyHeight"]
        x_center = x_min + w / 2
        y_center = y_min + h / 2

        label = [
            x_center / IMG_WIDTH,
            y_center / IMG_HEIGHT,
            w / IMG_WIDTH,
            h / IMG_HEIGHT,
            1.0
        ]

        # Original
        X.append(img)
        y.append(label)
        paths.append(image_path)

        # Flipped images (4 variations)
        for flipCode in [0, 1, -1]:
            flipped = cv2.flip(img, flipCode)
            label_flipped = label.copy()
            if flipCode == 1:
                label_flipped[0] = 1.0 - label_flipped[0]
            elif flipCode == 0:
                label_flipped[1] = 1.0 - label_flipped[1]
            elif flipCode == -1:
                label_flipped[0] = 1.0 - label_flipped[0]
                label_flipped[1] = 1.0 - label_flipped[1]
            X.append(flipped)
            y.append(label_flipped)
            paths.append(image_path + f"_flip{flipCode}")

    return np.array(X), np.array(y), paths

# === SIMPLE CNN DETECTION MODEL ===
def build_standard_detector():
    model = tf.keras.Sequential([
        layers.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='sigmoid')
    ])
    return model

# === METRICS ===
def compute_iou(box1, box2):
    x1 = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
    y1 = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    x2 = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)
    y2 = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# === VISUALIZATION ===
def visualize_prediction(image_path, prediction):
    x, y, w, h, conf = prediction
    x *= IMG_WIDTH
    y *= IMG_HEIGHT
    w *= IMG_WIDTH
    h *= IMG_HEIGHT

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    img = cv2.imread(image_path)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    print("Loading data...")

    X_total, y_total, image_paths_total = [], [], []

    for annotation_file, image_dir in ANNOTATION_FILES:
        X_part, y_part, paths_part = load_annotations(annotation_file, image_dir, (TARGET_WIDTH, TARGET_HEIGHT))
        X_total.append(X_part)
        y_total.append(y_part)
        image_paths_total.extend(paths_part)

    X = np.concatenate(X_total, axis=0)
    y = np.concatenate(y_total, axis=0)

    print("Building and training model...")
    model = build_standard_detector()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, batch_size=8, epochs=10, validation_split=0.1)

    print("Evaluating model...")
    y_pred = model.predict(X)

    total_iou = 0
    correct = 0
    for i in range(len(y)):
        iou = compute_iou(y_pred[i][:4], y[i][:4])
        total_iou += iou
        if iou > 0.5:
            correct += 1

    avg_iou = total_iou / len(y)
    accuracy = correct / len(y)
    print(f"MSE: {np.mean(np.square(y - y_pred)):.4f}, Avg IoU: {avg_iou:.4f}, Accuracy (IoU > 0.5): {accuracy*100:.2f}%")

    print("Testing on sample image...")
    sample_idx = 0
    pred = model.predict(X[sample_idx:sample_idx+1])[0]
    image_path = image_paths_total[sample_idx]
    visualize_prediction(image_path, pred)
