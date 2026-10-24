from pathlib import Path
import random
import cv2
import numpy as np
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_TRAIN = BASE_DIR / "datasets" / "visdrone_person" / "images" / "train"
SOURCE_VAL = BASE_DIR / "datasets" / "visdrone_person" / "images" / "val"
OUTPUT_DIR = BASE_DIR / "datasets" / "condition_classifier"

CLASSES = ["clear", "fog", "rain", "low_light"]
TRAIN_LIMIT = 1200
VAL_LIMIT = 200
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def make_fog(image):
    height, width = image.shape[:2]
    haze = np.full((height, width, 3), 235, dtype=np.uint8)
    return cv2.addWeighted(image, 0.62, haze, 0.38, 0)


def make_rain(image):
    result = image.copy()
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)
    random.seed(42)
    for _ in range(max(100, (height * width) // 1200)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        length = random.randint(8, 20)
        cv2.line(overlay, (x, y), (min(width - 1, x + 3), min(height - 1, y + length)), (210, 210, 210), 1)
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    return cv2.addWeighted(result, 0.88, overlay, 0.65, 0)


def make_low_light(image):
    darker = cv2.convertScaleAbs(image, alpha=0.38, beta=0)
    return darker


def transform(image, label):
    if label == "fog":
        return make_fog(image)
    if label == "rain":
        return make_rain(image)
    if label == "low_light":
        return make_low_light(image)
    return image


def prepare_split(source_dir, split, limit):
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Could not find {source_dir}. Make sure the VisDrone person dataset is prepared."
        )

    files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No images found in {source_dir}")

    random.Random(42).shuffle(files)
    files = files[:min(limit, len(files))]

    for label in CLASSES:
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        for label in CLASSES:
            converted = transform(image, label)
            target = OUTPUT_DIR / split / label / f"{path.stem}_{index}.jpg"
            if not cv2.imwrite(str(target), converted):
                raise RuntimeError(f"Could not write {target}")

    print(f"{split}: generated {len(files)} source images across {len(CLASSES)} classes.")


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    prepare_split(SOURCE_TRAIN, "train", TRAIN_LIMIT)
    prepare_split(SOURCE_VAL, "val", VAL_LIMIT)
    print(f"Dataset ready: {OUTPUT_DIR}")
    print("Important: fog, rain, and low-light examples are synthetic, not real weather images.")


if __name__ == "__main__":
    main()
