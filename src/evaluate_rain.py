
import cv2
import csv
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "datasets" / "Rain100L" / "archive" / "Rain100L" / "input"
TARGET_DIR = BASE_DIR / "datasets" / "Rain100L" / "archive" / "Rain100L" / "target"
OUTPUT_DIR = BASE_DIR / "results" / "rain_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

input_files = sorted(INPUT_DIR.glob("*"))
rows = []

for input_path in input_files:
    if not input_path.is_file():
        continue

    target_path = TARGET_DIR / input_path.name

    if not target_path.exists():
        print("Missing target:", input_path.name)
        continue

    rainy = cv2.imread(str(input_path))
    clean = cv2.imread(str(target_path))

    if rainy is None or clean is None:
        print("Could not read:", input_path.name)
        continue

    if rainy.shape != clean.shape:
        print("Size mismatch:", input_path.name)
        continue

    enhanced = apply_clahe(rainy)

    rainy_psnr = peak_signal_noise_ratio(clean, rainy, data_range=255)
    enhanced_psnr = peak_signal_noise_ratio(clean, enhanced, data_range=255)

    rainy_ssim = structural_similarity(
        clean, rainy, channel_axis=2, data_range=255
    )
    enhanced_ssim = structural_similarity(
        clean, enhanced, channel_axis=2, data_range=255
    )

    rows.append([
        input_path.name,
        round(rainy_psnr, 4),
        round(enhanced_psnr, 4),
        round(rainy_ssim, 4),
        round(enhanced_ssim, 4)
    ])

    if len(rows) % 20 == 0:
        print(f"Evaluated {len(rows)} image pairs")

csv_path = OUTPUT_DIR / "rain_metrics.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image",
        "Original_PSNR",
        "CLAHE_PSNR",
        "Original_SSIM",
        "CLAHE_SSIM"
    ])
    writer.writerows(rows)

if rows:
    count = len(rows)
    print("\nImage pairs evaluated:", count)
    print("Average original PSNR:", round(sum(r[1] for r in rows) / count, 4))
    print("Average CLAHE PSNR:", round(sum(r[2] for r in rows) / count, 4))
    print("Average original SSIM:", round(sum(r[3] for r in rows) / count, 4))
    print("Average CLAHE SSIM:", round(sum(r[4] for r in rows) / count, 4))
    print("Results saved to:", csv_path)
else:
    print("No valid image pairs were evaluated.")
