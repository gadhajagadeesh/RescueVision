from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Image to test
image_path = "sample_inputs/aerial_foggy.jpg"

# Run person detection
results = model.predict(
    source=image_path,
    classes=[0],       # COCO class 0 = person
    conf=0.25,
    save=True,
    project="results",
    name="foggy_detection"
)

# Count detected persons
person_count = 0

for result in results:
    for box in result.boxes:
        person_count += 1

        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print(f"\nPerson {person_count}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding box: {coordinates}")

print(f"\nTotal persons detected: {person_count}")
print("Detection completed!")