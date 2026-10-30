# RescueVision: Live Demonstration Guide

This guide provides a step-by-step walkthrough for demonstrating the **RescueVision** deep learning system during project reviews or presentations.

---

## 1. Quick Launch

Open PowerShell in the project directory (`C:\Users\gadha\OneDrive\Desktop\DL Project`):

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Streamlit will open the web interface in your default browser at `http://localhost:8501`.

---

## 2. Recommended Demonstration Sequence

Follow this 4-step sequence to showcase all key project contributions:

### Demo 1: Aerial Crowd Detection (Fine-Tuned VisDrone vs. Baseline)
* **Goal:** Demonstrate that fine-tuning on aerial drone viewpoints significantly outperforms generic ground detectors.
* **Test Image:** `sample_inputs/image1.jpeg`
* **Settings:**
  1. Image Processing Mode: `Auto (Select based on predicted condition)`
  2. Confidence Threshold: `0.25`
* **Steps:**
  1. Upload `sample_inputs/image1.jpeg`.
  2. Notice the condition classifier predicts **Clear (89.9%)** and passes the image through unmodified.
  3. Under **Person Detector Model**, select **`Pretrained YOLO11n (COCO Baseline)`**:
     - *Result:* Detects **20 people**.
  4. Now switch the detector to **`Fine-tuned VisDrone Model (Aerial-specific)`**:
     - *Result:* Detects **43 people**! (More than **2x** the detections of the baseline).
* **Talking Point:** Generic COCO detectors struggle with small aerial scales and top-down angles. Fine-tuning for 20 epochs on VisDrone more than doubled person detection coverage.

---

### Demo 2: Adverse Fog & Automatic Contrast Restoration
* **Goal:** Demonstrate automatic weather classification and condition-specific contrast recovery.
* **Test Image:** `sample_inputs/aerial_foggy.jpg`
* **Settings:**
  1. Person Detector Model: `Pretrained YOLO11n (COCO Baseline)`
  2. Image Processing Mode: `Auto (Select based on predicted condition)`
  3. Confidence Threshold: `0.25`
* **Steps:**
  1. Upload `sample_inputs/aerial_foggy.jpg`.
  2. **Automatic Detection:** The classifier immediately flags **Predicted Condition: Fog (100.0%)**.
  3. **Automatic Routing:** The app automatically applies **CLAHE local contrast boost** without manual intervention.
  4. **Side-by-Side Comparison:** 
     - Original Foggy: 2 people detected (confidences: `0.836`, `0.777`).
     - Processed (CLAHE): 2 people detected with **higher confidence** (`0.867`, `0.861`).
* **Talking Point:** The system automatically identifies atmospheric haze and applies local contrast equalisation, improving detection confidence on hazy victims.

---

### Demo 3: Low-Light Night Imagery & Gamma Brightening
* **Goal:** Showcase shadow recovery on dark scenes.
* **Test Image:** `sample_inputs/sample_low_light.png`
* **Settings:**
  1. Image Processing Mode: `Auto (Select based on predicted condition)`
* **Steps:**
  1. Upload `sample_inputs/sample_low_light.png`.
  2. **Automatic Detection:** The classifier predicts **Low light (100.0%)**.
  3. **Automatic Routing:** The app automatically applies **Gamma 1.8 brightening + CLAHE contrast boost**.
  4. Inspect Column 2: Notice how dark shadow details become clearly visible in the processed column.
* **Talking Point:** Low light requires non-linear gamma expansion rather than linear contrast stretching to recover details without clipping highlights.

---

### Demo 4: Rain Imagery & Edge-Preserving Streak Smoothing
* **Goal:** Demonstrate rain streak suppression.
* **Test Image:** `sample_inputs/sample_rain.png`
* **Settings:**
  1. Image Processing Mode: `Auto (Select based on predicted condition)`
* **Steps:**
  1. Upload `sample_inputs/sample_rain.png`.
  2. **Automatic Detection:** The classifier predicts **Rain (99.9%)**.
  3. **Automatic Routing:** The app applies **Bilateral streak smoothing + edge recovery**.
  4. Observe the reduction in high-frequency rain streaks while maintaining object outlines.
* **Talking Point:** Classical contrast enhancement (CLAHE) was proven to degrade rainy images (-7.59 dB PSNR on Rain100L); bilateral edge-preserving smoothing is essential to avoid amplifying streak noise.

---

## 3. Summary of Available Sample Images

| Filename | Weather Condition | Best Detector | Recommended Threshold |
|---|---|---|---|
| `sample_inputs/image1.jpeg` | Clear (Aerial crowd) | Fine-tuned VisDrone | `0.25` (43 detections) or `0.15` (63 detections) |
| `sample_inputs/aerial_foggy.jpg` | Fog | Pretrained YOLO11n | `0.25` (2 detections at 0.867, 0.861) |
| `sample_inputs/aerial_original.png` | Clear (Overhead 2 people) | Pretrained YOLO11n | `0.25` (2 detections at 0.878, 0.819) |
| `sample_inputs/sample_low_light.png` | Low Light | Pretrained / VisDrone | `0.25` |
| `sample_inputs/sample_rain.png` | Rain | Pretrained / VisDrone | `0.25` |
| `sample_inputs/test.jpeg` | Clear | Pretrained / VisDrone | `0.25` (3 detections) |

---

## 4. Known Academic Limitations & Honest Context

1. **Classical OpenCV vs. Deep Learning Restoration:**
   - The condition classifier is a trained deep neural network (`YOLO11n-cls`).
   - The person detector is a trained deep neural network (`YOLO11n` fine-tuned for 20 epochs).
   - The image-processing filters (CLAHE, Gamma, Bilateral) are **classical OpenCV heuristics**, not deep learning generative/restoration models (e.g., Restormer or Zero-DCE).
2. **Detection Count vs. Ground Truth:**
   - A higher detection count in the UI does not automatically guarantee higher accuracy. Verified Precision, Recall, and F1-score benchmarks are documented in `results/evaluation_summary.md`.
3. **VisDrone Model Confidence Calibration:**
   - The fine-tuned VisDrone model was trained on high-altitude datasets with tiny targets in clusters. On medium-altitude single-person shots, lowering the slider to `0.15 - 0.20` captures lower-confidence candidates.
