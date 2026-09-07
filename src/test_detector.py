from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect only persons in the aerial image
results = model.predict(
    source="sample_inputs/1080.avif",
    classes=[0],
    conf=0.25,
    save=True,
    project="results",
    name="aerial_detection"
)

# Print details of each detected person
for result in results:
    for box in result.boxes:
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print("\nDetected: Person")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding box: {coordinates}")

print("\nAerial person detection completed!")