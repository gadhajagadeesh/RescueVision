from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "datasets" / "condition_classifier"
OUTPUT_DIR = BASE_DIR / "runs" / "condition_classifier"


def main():
    if not (DATASET_DIR / "train").exists() or not (DATASET_DIR / "val").exists():
        raise FileNotFoundError(
            "Condition dataset not found. First run: python src/prepare_condition_dataset.py"
        )

    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(DATASET_DIR),
        epochs=15,
        imgsz=224,
        batch=16,
        patience=4,
        project=str(OUTPUT_DIR),
        name="yolo11n_condition",
        exist_ok=True
    )
    best_model = OUTPUT_DIR / "yolo11n_condition" / "weights" / "best.pt"
    print(f"Training finished. Best model should be at: {best_model}")
    print("Check the validation accuracy before relying on automatic condition predictions.")


if __name__ == "__main__":
    main()
