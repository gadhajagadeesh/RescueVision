from pathlib import Path
import cv2
import shutil

# Original VisDrone training dataset
dataset_root = Path(
    "datasets/VisDrone2019-DET-train/VisDrone2019-DET-train"
)

images_dir = dataset_root / "images"
annotations_dir = dataset_root / "annotations"

# New YOLO person-only dataset
output_root = Path("datasets/visdrone_person")

output_images = output_root / "images" / "train"
output_labels = output_root / "labels" / "train"

output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

# VisDrone:
# 1 = pedestrian
# 2 = people
PERSON_CLASSES = {1, 2}

processed_images = 0
converted_persons = 0

for image_path in images_dir.glob("*.jpg"):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image_height, image_width = image.shape[:2]

    annotation_path = annotations_dir / f"{image_path.stem}.txt"

    if not annotation_path.exists():
        continue

    yolo_lines = []

    with open(annotation_path, "r") as file:

        for line in file:

            values = line.strip().split(",")

            if len(values) < 8:
                continue

            x = float(values[0])
            y = float(values[1])
            width = float(values[2])
            height = float(values[3])
            category = int(values[5])

            # Keep only pedestrian and people
            if category not in PERSON_CLASSES:
                continue

            # Convert to YOLO format
            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height

            normalized_width = width / image_width
            normalized_height = height / image_height

            # YOLO class 0 = person
            yolo_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{normalized_width:.6f} {normalized_height:.6f}"
            )

            converted_persons += 1

    # Keep image only if it contains a person
    if yolo_lines:

        # Copy image
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        # Create YOLO label
        label_path = output_labels / f"{image_path.stem}.txt"

        with open(label_path, "w") as file:
            file.write("\n".join(yolo_lines))

        processed_images += 1


print("\nConversion completed!")
print("Images containing people:", processed_images)
print("Person annotations converted:", converted_persons)