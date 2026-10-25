import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# Project output folder
output_folder = "results/comparison"
os.makedirs(output_folder, exist_ok=True)

# Load the existing YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

# Read the original aerial image
image = cv2.imread("sample_inputs/aerial_original.png")

if image is None:
    raise FileNotFoundError(
        "Could not find sample_inputs/aerial_original.png"
    )

# Create a simulated foggy image
white_layer = np.full_like(image, 255)
foggy_image = cv2.addWeighted(
    image, 0.55, white_layer, 0.45, 0
)

# Apply CLAHE to the foggy image
lab_image = cv2.cvtColor(foggy_image, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_image)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
enhanced_l = clahe.apply(l_channel)

enhanced_lab = cv2.merge(
    (enhanced_l, a_channel, b_channel)
)
enhanced_image = cv2.cvtColor(
    enhanced_lab, cv2.COLOR_LAB2BGR
)

# Test all three image conditions
conditions = [
    ("clear", image),
    ("foggy", foggy_image),
    ("clahe_enhanced", enhanced_image)
]

csv_path = os.path.join(
    output_folder, "detection_counts.csv"
)

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Condition",
        "Detected_Persons",
        "Confidence_Scores"
    ])

    for name, current_image in conditions:
        print("\nTesting:", name)

        # Save the input image
        input_path = os.path.join(
            output_folder, name + ".jpg"
        )
        cv2.imwrite(input_path, current_image)

        # Detect people (COCO class 0 = person)
        results = model.predict(
            source=current_image,
            classes=[0],
            conf=0.25,
            verbose=False
        )

        result = results[0]
        count = len(result.boxes)

        scores = [
            round(float(box.conf[0]), 3)
            for box in result.boxes
        ]

        # Save the image with bounding boxes
        boxed_image = result.plot()
        boxed_path = os.path.join(
            output_folder, name + "_detection.jpg"
        )
        cv2.imwrite(boxed_path, boxed_image)

        writer.writerow([name, count, scores])
        print("Detected persons:", count)
        print("Confidence scores:", scores)
        print("Saved:", boxed_path)

print("\nComparison completed successfully!")
print("Output folder:", output_folder)
print("CSV file:", csv_path)
