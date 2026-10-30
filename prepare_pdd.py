from pathlib import Path
import random
import shutil

source = Path(r"C:\Users\gadha\OneDrive\Desktop\DL Project\Post-Disaster-Dataset-main\879real survivor")
destination = Path(r"C:\Users\gadha\OneDrive\Desktop\DL Project\datasets\PDD_YOLO")

random.seed(42)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(p for p in (source / "images").iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions)

random.shuffle(images)

train_end = int(len(images) * 0.70)
val_end = train_end + int(len(images) * 0.20)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

for split, files in splits.items():
    image_dir = destination / "images" / split
    label_dir = destination / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image in files:
        label = source / "labels" / (image.stem + ".txt")
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

yaml = f"""path: '{destination.as_posix()}'
train: images/train
val: images/val
test: images/test

names:
  0: person
"""

(destination / "pdd.yaml").write_text(yaml, encoding="utf-8")

print("Dataset prepared successfully!")
for split, files in splits.items():
    print(f"{split}: {len(files)} images")
print("Configuration:", destination / "pdd.yaml")
