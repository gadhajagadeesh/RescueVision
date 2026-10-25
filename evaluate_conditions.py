from ultralytics import YOLO
import cv2
import os
import glob
import csv
import numpy as np

model = YOLO("models/visdrone_person_20epochs_best.pt")

image_dir = "datasets/visdrone_person/images/val"
label_dir = "datasets/visdrone_person/labels/val"
output_dir = "results/condition_evaluation"
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

if not image_paths:
    raise FileNotFoundError("No validation images found.")

conditions = ["clear", "foggy", "clahe_enhanced"]
counts = {c: {"TP": 0, "FP": 0, "FN": 0} for c in conditions}

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    return intersection / (area1 + area2 - intersection + 1e-9)

for index, image_path in enumerate(image_paths, 1):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, image_name + ".txt")

    image = cv2.imread(image_path)
    if image is None:
        continue

    height, width = image.shape[:2]

    ground_truth = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                values = line.split()
                if len(values) < 5 or int(values[0]) != 0:
                    continue

                _, xc, yc, bw, bh = map(float, values[:5])
                ground_truth.append([
                    (xc - bw/2) * width,
                    (yc - bh/2) * height,
                    (xc + bw/2) * width,
                    (yc + bh/2) * height
                ])

    white = np.full_like(image, 255)
    foggy = cv2.addWeighted(image, 0.55, white, 0.45, 0)

    lab = cv2.cvtColor(foggy, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR
    )

    variants = {
        "clear": image,
        "foggy": foggy,
        "clahe_enhanced": enhanced
    }

    for condition, variant in variants.items():
        result = model.predict(
            variant, conf=0.25, iou=0.7, verbose=False
        )[0]

        predictions = []
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                predictions.append([
                    float(v) for v in box.xyxy[0].cpu().tolist()
                ])

        pairs = []
        for pi, pred in enumerate(predictions):
            for gi, gt in enumerate(ground_truth):
                iou_value = calculate_iou(pred, gt)
                if iou_value >= 0.5:
                    pairs.append((iou_value, pi, gi))

        pairs.sort(reverse=True)
        matched_preds = set()
        matched_gts = set()

        for _, pi, gi in pairs:
            if pi not in matched_preds and gi not in matched_gts:
                matched_preds.add(pi)
                matched_gts.add(gi)

        counts[condition]["TP"] += len(matched_preds)
        counts[condition]["FP"] += len(predictions) - len(matched_preds)
        counts[condition]["FN"] += len(ground_truth) - len(matched_gts)

    if index % 50 == 0:
        print(f"Processed {index}/{len(image_paths)} images")

csv_path = os.path.join(output_dir, "condition_metrics.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Condition", "TP", "FP", "FN", "Precision", "Recall", "F1"])

    for condition, values in counts.items():
        tp, fp, fn = values["TP"], values["FP"], values["FN"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        writer.writerow([
            condition, tp, fp, fn,
            round(precision, 4), round(recall, 4), round(f1, 4)
        ])

        print(
            f"{condition}: TP={tp}, FP={fp}, FN={fn}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

print(f"\nMetrics saved to {csv_path}")
