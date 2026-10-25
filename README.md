# RescueVision

RescueVision is a deep learning project for improving victim/person detection from drone imagery captured under adverse visibility conditions such as fog, rain, and low light.

## Project Idea

The planned pipeline is:

Drone Image/Video  
→ Identify visibility/degradation condition  
→ Apply condition-specific image enhancement/restoration  
→ Detect persons/victims  
→ Compare detection performance before and after enhancement

The main aim is to study whether improving image visibility can improve downstream person/victim detection in difficult disaster conditions.

## Current Implementation Status

The following work has been completed so far:

- Set up the project environment and required Python libraries.
- Tested pretrained YOLO11n for person detection.
- Downloaded and explored the VisDrone dataset.
- Converted VisDrone annotations into YOLO format for person-only detection.
- Prepared training and validation datasets.
- Verified converted YOLO bounding-box annotations.
- Evaluated the pretrained YOLO11n model on the VisDrone validation set.
- Started fine-tuning YOLO11n on the VisDrone person dataset using Google Colab GPU.
- Created an initial synthetic fog image for later raw-vs-enhanced experiments.

## Baseline Result

Pretrained YOLO11n was evaluated on the VisDrone person validation set before fine-tuning.

| Metric | Result |
|---|---:|
| Precision | 0.437 |
| Recall | 0.181 |
| mAP@50 | 0.160 |
| mAP@50-95 | 0.0574 |

These values serve as the initial baseline for later comparison with the fine-tuned model and the image-enhancement pipeline.

## Dataset

Current detection dataset:

**VisDrone2019-DET**

For the current experiment, person-related annotations were extracted and converted into a single YOLO class:

`0: person`

The complete dataset is not stored in this GitHub repository because of its size.

## Project Structure

```text
RescueVision/
├── config/
│   └── visdrone.yaml
├── sample_inputs/
├── src/
│   ├── check_dataset.py
│   ├── check_yolo_labels.py
│   ├── convert_visdrone.py
│   ├── convert_visdrone_val.py
│   ├── create_fog.py
│   ├── detect_person.py
│   ├── inspect_visdrone.py
│   └── test_detector.py
├── .gitignore
└── README.md



## Setup and Run Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/gadhajagadeesh/RescueVision.git
cd RescueVision
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to use the RescueVision application.

## Important Notes

- The datasets and trained model weights are not included in this repository.
- The required model files must be obtained separately and placed in the paths expected by the application.
- Detection results depend on the model used, confidence threshold, and input image.
- Image enhancement methods are experimental; their effectiveness must be evaluated on the relevant datasets.

## Team

- Gadha J
- Hari Krishna
- Upasana
