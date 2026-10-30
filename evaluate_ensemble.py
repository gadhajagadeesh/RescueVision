
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "PDD_YOLO"

MODELS = {
    "VisDrone": BASE / "models" / "visdrone_person_20epochs_best.pt",
    "PDD": BASE / "models" / "yolov8n_pdd_best.pt",
    "COCO": BASE / "yolo11n.pt"
}

IMAGE_DIR = DATA / "images" / "test"
LABEL_DIR = DATA / "labels" / "test"

CONF = 0.25
IOU_MATCH = 0.50


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def get_predictions(model, image):
    result = model.predict(
        image,
        conf=CONF,
        imgsz=1280,
        classes=[0],
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:
        predictions.append({
            "box": box.xyxy[0].cpu().numpy().tolist(),
            "score": float(box.conf[0].item()),
            "model": None
        })

    return predictions


def make_ensemble(predictions):
    predictions.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    used = set()
    final = []

    for i, pred in enumerate(predictions):
        if i in used:
            continue

        group = [pred]
        used.add(i)

        for j in range(i + 1, len(predictions)):
            if j in used:
                continue

            other = predictions[j]

            if other["model"] in {
                item["model"] for item in group
            }:
                continue

            if iou(pred["box"], other["box"]) >= IOU_MATCH:
                group.append(other)
                used.add(j)

        if len({x["model"] for x in group}) >= 2:
            boxes = np.array([x["box"] for x in group])
            scores = np.array([x["score"] for x in group])

            weights = scores / scores.sum()
            final_box = np.average(
                boxes,
                axis=0,
                weights=weights
            )

            final.append(final_box.tolist())

    return final


def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id, xc, yc, w, h = map(float, values)

        if int(class_id) != 0:
            continue

        boxes.append([
            (xc - w/2) * width,
            (yc - h/2) * height,
            (xc + w/2) * width,
            (yc + h/2) * height
        ])

    return boxes


def evaluate(predictions_by_image, ground_truth_by_image):
    tp = fp = fn = 0

    for predictions, ground_truth in zip(
        predictions_by_image,
        ground_truth_by_image
    ):
        matched = set()

        for pred in predictions:
            best_iou = 0
            best_index = -1

            for index, gt in enumerate(ground_truth):
                if index in matched:
                    continue

                score = iou(pred, gt)

                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_iou >= 0.50:
                tp += 1
                matched.add(best_index)
            else:
                fp += 1

        fn += len(ground_truth) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return tp, fp, fn, precision, recall, f1


print("Loading all three models...")

models = {
    name: YOLO(str(path))
    for name, path in MODELS.items()
}

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

all_gt = []
model_predictions = {name: [] for name in models}
ensemble_predictions = []

print("Evaluating", len(image_files), "test images...")

for index, image_path in enumerate(image_files, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    label_path = LABEL_DIR / (image_path.stem + ".txt")
    gt = load_ground_truth(label_path, width, height)
    all_gt.append(gt)

    combined = []

    for name, model in models.items():
        predictions = get_predictions(model, image)

        for pred in predictions:
            pred["model"] = name

        model_predictions[name].append(
            [p["box"] for p in predictions]
        )

        combined.extend(predictions)

    ensemble_predictions.append(make_ensemble(combined))

    print(f"Processed {index}/{len(image_files)}")

print("\n========== RESULTS ==========")

for name, predictions in model_predictions.items():
    tp, fp, fn, precision, recall, f1 = evaluate(
        predictions, all_gt
    )

    print(f"\n{name}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

tp, fp, fn, precision, recall, f1 = evaluate(
    ensemble_predictions, all_gt
)

print("\n========== 3-MODEL ENSEMBLE ==========")
print(f"TP: {tp} | FP: {fp} | FN: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nEvaluation completed.")
