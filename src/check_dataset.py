from ultralytics import YOLO

# Load pretrained YOLO
model = YOLO("yolo11n.pt")

# Check our validation dataset using the YAML file
metrics = model.val(
    data="config/visdrone.yaml",
    imgsz=640,
    batch=4,
    device="cpu"
)

print("\nDataset check completed!")