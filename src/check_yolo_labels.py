import cv2
from pathlib import Path

images_dir = Path("datasets/visdrone_person/images/train")
labels_dir = Path("datasets/visdrone_person/labels/train")

# Get first image
image_path = next(images_dir.glob("*.jpg"))

# Find matching YOLO label
label_path = labels_dir / f"{image_path.stem}.txt"

image = cv2.imread(str(image_path))

height, width = image.shape[:2]

with open(label_path, "r") as file:

    for line in file:

        values = line.strip().split()

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized YOLO coordinates back to pixels
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)

        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "person",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

# Save result
output_path = "results/check_visdrone_labels.jpg"

cv2.imwrite(output_path, image)

print("Checked image:", image_path.name)
print("Label file:", label_path.name)
print("Verification image saved to:", output_path)