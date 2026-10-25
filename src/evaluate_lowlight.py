
import cv2
import csv
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "LOL" / "archive (1)" / "LOLv1" / "test"
LOW_DIR = DATA_DIR / "low"
HIGH_DIR = DATA_DIR / "high"
OUTPUT_DIR = BASE_DIR / "results" / "lowlight_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


rows = []

for low_path in sorted(LOW_DIR.glob("*")):
    if not low_path.is_file():
        continue

    high_path = HIGH_DIR / low_path.name

    if not high_path.exists():
        print("Missing normal-light target:", low_path.name)
        continue

    low = cv2.imread(str(low_path))
    high = cv2.imread(str(high_path))

    if low is None or high is None:
        print("Could not read:", low_path.name)
        continue

    if low.shape != high.shape:
        print("Size mismatch:", low_path.name)
        continue

    enhanced = apply_clahe(low)

    low_psnr = peak_signal_noise_ratio(high, low, data_range=255)
    enhanced_psnr = peak_signal_noise_ratio(high, enhanced, data_range=255)

    low_ssim = structural_similarity(
        high, low, channel_axis=2, data_range=255
    )
    enhanced_ssim = structural_similarity(
        high, enhanced, channel_axis=2, data_range=255
    )

    rows.append([
        low_path.name,
        round(low_psnr, 4),
        round(enhanced_psnr, 4),
        round(low_ssim, 4),
        round(enhanced_ssim, 4)
    ])

csv_path = OUTPUT_DIR / "lowlight_metrics.csv"

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
    print("Image pairs evaluated:", count)
    print("Average original PSNR:", round(sum(r[1] for r in rows) / count, 4))
    print("Average CLAHE PSNR:", round(sum(r[2] for r in rows) / count, 4))
    print("Average original SSIM:", round(sum(r[3] for r in rows) / count, 4))
    print("Average CLAHE SSIM:", round(sum(r[4] for r in rows) / count, 4))
    print("Results saved to:", csv_path)
else:
    print("No valid image pairs were evaluated.")
