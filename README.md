# RescueVision: Drone Person Detection Under Adverse Conditions

**RescueVision** is a deep learning and computer vision framework for improving victim and person detection in drone/aerial imagery captured during disaster response scenarios under adverse visibility conditions (such as fog, low light, and rain).

---

## 1. Project Overview & Pipeline

In search-and-rescue operations, drones provide vital situational awareness. However, small victim sizes, top-down perspectives, and poor visibility (fog, darkness, rain) cause standard object detectors to fail.

RescueVision addresses this challenge via an end-to-end multi-stage pipeline:

```text
[Input Drone Image / Video]
           │
           ▼
[Stage 1: Condition Classifier] (Trained YOLO11n-cls)
     ├─ Clear
     ├─ Fog
     ├─ Low Light
     └─ Rain
           │
           ▼
[Stage 2: Condition-Specific Processing] (OpenCV Heuristics)
     ├─ Clear     -> Pass-through (no modification)
     ├─ Fog       -> CLAHE local contrast enhancement
     ├─ Low Light -> Non-linear Gamma brightening + CLAHE
     └─ Rain      -> Bilateral filter streak smoothing + edge recovery
           │
           ▼
[Stage 3: Drone Person Detection] (Fine-Tuned YOLO11n)
     ├─ Pretrained YOLO11n Baseline (COCO)
     └─ Fine-Tuned VisDrone Detector (Trained on aerial viewpoints)
           │
           ▼
[Stage 4: Side-by-Side Comparison & Metric Analysis]
     ├─ Detection count and average confidence
     └─ Ground-truth evaluation (Precision, Recall, F1-Score)
```

---

## 2. Models & Artifacts

| Component | Architecture / Method | Weights / Script | Dataset |
|---|---|---|---|
| **Aerial Person Detector** | YOLO11n (Fine-tuned, 20 epochs) | `models/visdrone_person_20epochs_best.pt` | VisDrone2019-DET (`0: person`) |
| **Baseline Detector** | Pretrained YOLO11n | `yolo11n.pt` | MS-COCO (`0: person`) |
| **Condition Classifier** | YOLO11n-cls (Trained, 15 epochs) | `models/condition_classifier_best.pt` | Synthetic balanced split (4 classes) |
| **Image Processing** | Classical OpenCV heuristics | `app.py` | CLAHE, Gamma, Bilateral Filter |

---

## 3. Real Measured Experimental Results

All numbers reported below are measured values from the project's evaluation scripts and saved result CSVs.

### A. Pretrained Baseline vs. Fine-Tuned VisDrone Detector

Evaluated on the **VisDrone2019-DET validation split** (531 images, 13,969 person ground-truth instances):

| Model | Weights | Precision | Recall | mAP@50 | mAP@50-95 | F1-Score |
|---|---|---|---|---|---|---|
| **Pretrained YOLO11n (COCO)** | `yolo11n.pt` | 0.4370 | 0.1810 | 0.1600 | 0.0574 | ~0.256 |
| **Fine-Tuned VisDrone Detector** | `visdrone_person_20epochs_best.pt` | **0.5811** | **0.4279** | — | — | **0.4929** |

*Key finding:* Fine-tuning specifically on aerial viewpoints increased Precision from **43.7% to 58.1%** (+14.4%) and more than doubled Recall from **18.1% to 42.8%** (+24.7%).

### B. Adverse Weather Person Detection Evaluation

Evaluated with `evaluate_conditions.py` on 531 VisDrone validation images at confidence threshold 0.25 and IoU threshold 0.50:

| Condition / Variant | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Clear (Clean Drone Images)** | 5,978 | 4,309 | 7,991 | **0.5811** | **0.4279** | **0.4929** |
| **Simulated Fog (Raw Adverse)** | 5,326 | 3,749 | 8,643 | 0.5869 | 0.3813 | 0.4622 |
| **CLAHE Enhanced Fog** | 5,258 | 3,255 | 8,711 | **0.6176** | 0.3764 | **0.4678** |

*Key finding:* Simulated fog noticeably dropped detection Recall and F1. CLAHE reduced false positive detections, boosting Precision to **61.76%** and raising the overall F1-score from **0.4622 to 0.4678**.

### C. Condition Classifier Performance

Trained for 15 epochs on a 4-class balanced adverse weather dataset (`runs/condition_classifier/yolo11n_condition/results.csv`):

| Metric | Result |
|---|---|
| **Validation Top-1 Accuracy** | **99.63%** (peak **99.88%**) |
| **Validation Top-5 Accuracy** | **100.00%** |
| **Validation Loss** | **0.0372** |

### D. Image Quality Benchmarking (PSNR & SSIM)

Evaluated on standard vision restoration test sets:

* **Low-Light (LOL Dataset, 15 test pairs via `src/evaluate_lowlight.py`):**
  - Original Low-Light: PSNR = **7.77 dB**, SSIM = **0.1914**
  - CLAHE Enhanced: PSNR = **9.13 dB**, SSIM = **0.3565** (+1.36 dB PSNR, +0.165 SSIM gain)
* **Rain (Rain100L Dataset, 100 test pairs via `src/evaluate_rain.py`):**
  - Original Rainy: PSNR = **25.52 dB**, SSIM = **0.8140**
  - CLAHE Enhanced: PSNR = **17.93 dB**, SSIM = **0.6353** (-7.59 dB degradation)
  - *Scientific Insight:* Generic contrast enhancement worsens rainy images by sharpening rain streak noise, demonstrating the strict necessity of edge-preserving streak filtering or dedicated deep deraining models.

---

## 4. Project Structure

```text
RescueVision/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Python dependencies
├── evaluate_conditions.py      # VisDrone weather detection evaluation script
├── config/
│   └── visdrone.yaml           # VisDrone YOLO configuration
├── models/
│   ├── visdrone_person_20epochs_best.pt  # Fine-tuned aerial person detector
│   └── condition_classifier_best.pt       # Trained 4-class condition classifier
├── results/
│   ├── evaluation_summary.md   # Detailed consolidated benchmark report
│   ├── baseline_notes.txt      # Initial YOLO11n baseline measurements
│   ├── condition_evaluation/   # VisDrone detection metrics (Clear vs Fog vs CLAHE)
│   ├── lowlight_evaluation/    # LOL dataset PSNR/SSIM evaluation
│   ├── rain_evaluation/        # Rain100L dataset PSNR/SSIM evaluation
│   └── comparison/             # Sample detection comparison logs
├── sample_inputs/              # Sample aerial images for demonstration
├── src/
│   ├── check_dataset.py        # Dataset structure validation
│   ├── check_yolo_labels.py    # YOLO label verification
│   ├── compare_conditions.py   # Condition comparison utility
│   ├── convert_visdrone.py     # VisDrone train annotation converter
│   ├── convert_visdrone_val.py # VisDrone val annotation converter
│   ├── create_fog.py           # Synthetic fog generation utility
│   ├── detect_person.py        # Single-image detection test script
│   ├── evaluate_lowlight.py    # LOL benchmark evaluation script
│   ├── evaluate_rain.py        # Rain100L benchmark evaluation script
│   ├── inspect_visdrone.py     # Annotation inspection script
│   ├── prepare_condition_dataset.py # Condition dataset preparation
│   ├── test_detector.py        # Quick detector inference test
│   └── train_condition_classifier.py # YOLO classification training script
└── README.md
```

---

## 5. Setup & Running the Application

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/gadhajagadeesh/RescueVision.git
cd RescueVision

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (typically `http://localhost:8501`).

### 3. Using the Web App
1. **Choose a Person Detector:** In the sidebar, select between the **Fine-tuned VisDrone Model** (recommended for drone footage) and the **Pretrained YOLO11n Baseline**.
2. **Select Image Processing Mode:**
   - **`Auto (Select based on predicted condition)`** *(Recommended)*: Automatically routes images through the appropriate processing method based on the condition classifier's prediction.
   - Or choose a manual mode (Pass-through, Fog CLAHE, Low-Light Gamma+CLAHE, Rain Bilateral, or Synthetic Fog Demo).
3. **Upload an Image:** Upload any aerial image from `sample_inputs/` (e.g. `aerial_original.png` or `aerial_foggy.jpg`).
4. **Inspect Side-by-Side Results:** View original vs. processed images, bounding-box annotations, detected person counts, individual and average confidence scores, and count difference metrics.

---

## 6. Academic Limitations & Future Work

1. **OpenCV Heuristics vs. Deep Restoration:** The image-enhancement methods in `app.py` use classical OpenCV techniques (CLAHE, gamma adjustment, bilateral filtering). While lightweight and fast, they are not dedicated deep learning restoration models (e.g., DehazeNet, Zero-DCE, or Restormer).
2. **Synthetic vs. Real Adverse Imagery:** The condition classifier was trained on synthetic weather variations. Testing on diverse, real-world adverse drone datasets is the recommended next step.
3. **Detection Count vs. Ground-Truth:** Higher detection count in the UI does not automatically imply better precision; ground-truth IoU evaluation remains the definitive measure of accuracy.
4. **Future Expansion:** Integration of end-to-end deep restoration backbones and real-time drone video stream processing.

---

## 7. Team

- **Gadha J**
- **Hari Krishna**
- **Upasana**
