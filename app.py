
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

VISDRONE_PATH = BASE_DIR / "models" / "visdrone_person_20epochs_best.pt"
PDD_PATH = BASE_DIR / "models" / "yolov8n_pdd_best.pt"
COCO_PATH = BASE_DIR / "yolo11n.pt"
CLASSIFIER_PATH = BASE_DIR / "models" / "condition_classifier_best.pt"

MODEL_PATHS = {
    "VisDrone": VISDRONE_PATH,
    "PDD": PDD_PATH,
    "COCO": COCO_PATH
}

st.set_page_config(
    page_title="RescueVision",
    page_icon="🚁",
    layout="wide"
)


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


def classify_condition(model, image):
    result = model.predict(
        source=image,
        imgsz=224,
        verbose=False
    )[0]

    class_id = result.probs.top1
    condition = result.names[class_id]
    confidence = float(result.probs.top1conf.item())

    return condition, confidence


# ---------------------------------------------------------
# Classical OpenCV Enhancement Methods
# ---------------------------------------------------------

def enhance_clear(image):
    return image.copy(), "Clear: Pass-through (no modification applied)"


def enhance_fog(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    ), "Fog: CLAHE local contrast boost (OpenCV)"


def enhance_low_light(image):
    inv_gamma = 1.0 / 1.8

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    gamma_corrected = cv2.LUT(image, table)

    lab_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    ), "Low Light: Gamma brightening + CLAHE (OpenCV)"


def enhance_rain(image):
    filtered = cv2.bilateralFilter(
        image,
        d=7,
        sigmaColor=45,
        sigmaSpace=45
    )

    blurred = cv2.GaussianBlur(filtered, (0, 0), 2.0)

    sharpened = cv2.addWeighted(
        filtered,
        1.2,
        blurred,
        -0.2,
        0
    )

    return sharpened, "Rain: Bilateral streak smoothing (OpenCV)"


def add_fog(image):
    white_layer = np.full_like(image, 255)

    return cv2.addWeighted(
        image,
        0.55,
        white_layer,
        0.45,
        0
    ), "Synthetic fog overlay (Demonstration)"


def get_auto_enhancement(condition_name, image):
    if condition_name == "clear":
        return enhance_clear(image)

    if condition_name == "fog":
        return enhance_fog(image)

    if condition_name == "low_light":
        return enhance_low_light(image)

    if condition_name == "rain":
        return enhance_rain(image)

    return enhance_clear(image)


def process_image(image, condition_option, predicted_condition=None):
    if condition_option.startswith("Auto"):
        if predicted_condition:
            return get_auto_enhancement(predicted_condition, image)

        return enhance_clear(image)

    if "Clear" in condition_option:
        return enhance_clear(image)

    if "Fog: CLAHE" in condition_option:
        return enhance_fog(image)

    if "Low Light" in condition_option:
        return enhance_low_light(image)

    if "Rain" in condition_option:
        return enhance_rain(image)

    if condition_option == "Synthetic fog (Degradation demo)":
        return add_fog(image)

    if condition_option == "Synthetic fog + CLAHE (Demo)":
        foggy, _ = add_fog(image)
        enhanced, _ = enhance_fog(foggy)
        return enhanced, "Synthetic fog + CLAHE (Demo)"

    return image.copy(), condition_option


# ---------------------------------------------------------
# Ensemble Detection
# ---------------------------------------------------------

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_model_predictions(model, image, confidence, model_name):
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=1280,
        classes=[0],
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:
        predictions.append({
            "box": box.xyxy[0].cpu().numpy().tolist(),
            "confidence": float(box.conf[0].item()),
            "model": model_name
        })

    return predictions


def ensemble_detection(image, confidence, iou_threshold=0.5):
    all_predictions = []

    # Run all three models
    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(model_path)

        predictions = get_model_predictions(
            model,
            image,
            confidence,
            model_name
        )

        all_predictions.extend(predictions)

    # Sort predictions by confidence
    all_predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    used = set()
    final_detections = []

    # Match overlapping predictions from different models
    for i, prediction in enumerate(all_predictions):

        if i in used:
            continue

        group = [prediction]
        used.add(i)

        models_in_group = {prediction["model"]}

        for j in range(i + 1, len(all_predictions)):

            if j in used:
                continue

            candidate = all_predictions[j]

            if candidate["model"] in models_in_group:
                continue

            overlap = calculate_iou(
                prediction["box"],
                candidate["box"]
            )

            if overlap >= iou_threshold:
                group.append(candidate)
                models_in_group.add(candidate["model"])
                used.add(j)

        # Require agreement from at least two models
        if len(models_in_group) >= 2:

            boxes = np.array([
                item["box"] for item in group
            ])

            scores = np.array([
                item["confidence"] for item in group
            ])

            # Confidence-weighted average bounding box
            weights = scores / scores.sum()
            final_box = np.average(
                boxes,
                axis=0,
                weights=weights
            )

            final_detections.append({
                "box": final_box.tolist(),
                "confidence": float(scores.mean()),
                "agreement": len(models_in_group),
                "models": list(models_in_group)
            })

    # Draw final ensemble detections
    annotated_image = image.copy()

    for detection in final_detections:

        x1, y1, x2, y2 = map(
            int,
            detection["box"]
        )

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"Person {detection['confidence']:.2f} "
            f"({detection['agreement']}/3)"
        )

        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated_image, final_detections, all_predictions


# ---------------------------------------------------------
# Model Benchmark Results
# ---------------------------------------------------------

BENCHMARK_RESULTS = {
    "PDD YOLOv8n": {
        "dataset": "PDD test set",
        "images": 89,
        "precision": 0.844,
        "recall": 0.769,
        "f1": 0.805,
        "map50": 0.860,
        "map5095": 0.597
    },
    "VisDrone YOLO11n": {
        "dataset": "VisDrone validation set",
        "images": 531,
        "precision": 0.5811,
        "recall": 0.4279,
        "f1": 0.4929,
        "map50": None,
        "map5095": None
    },
    "Pretrained YOLO11n": {
        "dataset": "VisDrone evaluation set",
        "images": None,
        "precision": 0.437,
        "recall": 0.181,
        "f1": 0.2559,
        "map50": 0.160,
        "map5095": 0.0574
    },
    "3-Model Ensemble": {
    "dataset": "PDD Test Set",
    "images": 89,
    "precision": 0.8541,
    "recall": 0.7425,
    "f1": 0.7944,
    "map50": None,
    "map5095": None
},
}


def display_percentage(value):
    if value is None:
        return "Not evaluated"
    return f"{value * 100:.2f}%"


def show_benchmark_dashboard():
    st.header("Model Performance Dashboard")

    st.write(
        "Benchmark results from the individual model evaluations. "
        "Results come from different datasets and evaluation runs, "
        "so they should not be treated as a direct model ranking."
    )

    selected_benchmark = st.selectbox(
        "Select model benchmark",
        options=list(BENCHMARK_RESULTS.keys())
    )

    result = BENCHMARK_RESULTS[selected_benchmark]

    st.caption(
        f"Evaluation dataset: {result['dataset']}"
    )

    if result["images"] is not None:
        st.caption(
            f"Images evaluated: {result['images']}"
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Precision",
        display_percentage(result["precision"])
    )

    col2.metric(
        "Recall",
        display_percentage(result["recall"])
    )

    col3.metric(
        "F1-score",
        display_percentage(result["f1"])
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "mAP@50",
        display_percentage(result["map50"])
    )

    col5.metric(
        "mAP@50–95",
        display_percentage(result["map5095"])
    )

    st.divider()

    st.subheader("Benchmark Comparison")

    comparison_rows = []

    for model_name, values in BENCHMARK_RESULTS.items():
        comparison_rows.append({
            "Model": model_name,
            "Dataset": values["dataset"],
            "Precision": display_percentage(values["precision"]),
            "Recall": display_percentage(values["recall"]),
            "F1-score": display_percentage(values["f1"]),
            "mAP@50": display_percentage(values["map50"]),
            "mAP@50–95": display_percentage(values["map5095"])
        })

    st.dataframe(
        comparison_rows,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The ensemble has not yet been evaluated against ground-truth "
        "annotations. Its metrics will be added after a separate test-set "
        "evaluation. Missing values are intentionally not estimated."
    )

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.title("🚁 RescueVision")
show_benchmark_dashboard()

st.divider()
st.subheader(
    "Disaster Drone Imagery: Condition Classification, "
    "Image Enhancement and Ensemble Person Detection"
)

st.write(
    "Upload an aerial image to identify visibility conditions, "
    "apply image enhancement and detect people using a three-model ensemble."
)


# Sidebar
st.sidebar.header("Ensemble Detection Settings")

st.sidebar.info(
    "All three person detectors are used together: "
    "VisDrone, PDD and COCO."
)

confidence = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.20,
    step=0.01
)

iou_threshold = st.sidebar.slider(
    "Bounding-box IoU threshold",
    min_value=0.30,
    max_value=0.80,
    value=0.50,
    step=0.05
)

PROCESSING_OPTIONS = [
    "Auto (Select based on predicted condition)",
    "Clear: Pass-through (no enhancement)",
    "Fog: CLAHE contrast boost",
    "Low Light: Gamma correction + CLAHE",
    "Rain: Bilateral streak smoothing",
    "Synthetic fog (Degradation demo)",
    "Synthetic fog + CLAHE (Demo)"
]

selected_processing = st.sidebar.selectbox(
    "Image Processing Mode",
    options=PROCESSING_OPTIONS,
    index=0
)

st.sidebar.caption(
    "The ensemble accepts a detection when at least two different "
    "models predict a person in overlapping bounding boxes."
)


# Check model files
missing_models = [
    name
    for name, path in MODEL_PATHS.items()
    if not path.exists()
]

if missing_models:
    st.error(
        "Missing model files: " + ", ".join(missing_models)
    )
    st.stop()

if not CLASSIFIER_PATH.exists():
    st.warning(
        "Condition classifier is missing. "
        "Automatic condition prediction will be disabled."
    )


st.caption(
    "Image enhancement uses classical OpenCV methods, not deep-learning "
    "restoration. The ensemble combines predictions from three "
    "individually trained or pretrained YOLO detectors."
)


# Upload
uploaded_file = st.file_uploader(
    "Upload a drone or aerial image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")

        original_rgb = np.array(pil_image)

        original_bgr = cv2.cvtColor(
            original_rgb,
            cv2.COLOR_RGB2BGR
        )

        with st.spinner(
            "Classifying condition and running all three detection models..."
        ):

            predicted_condition = None
            condition_confidence = None

            # Condition classification
            if CLASSIFIER_PATH.exists():

                classifier = load_model(CLASSIFIER_PATH)

                predicted_condition, condition_confidence = classify_condition(
                    classifier,
                    original_bgr
                )

            # Enhancement
            processed_bgr, processing_description = process_image(
                original_bgr,
                selected_processing,
                predicted_condition
            )

            # Ensemble detection on original image
            original_annotated, original_detections, original_predictions = (
                ensemble_detection(
                    original_bgr,
                    confidence,
                    iou_threshold
                )
            )

            # Ensemble detection on processed image
            processed_annotated, processed_detections, processed_predictions = (
                ensemble_detection(
                    processed_bgr,
                    confidence,
                    iou_threshold
                )
            )


        # Condition result
        if predicted_condition is not None:

            condition_display_names = {
                "clear": "Clear",
                "fog": "Fog",
                "rain": "Rain",
                "low_light": "Low light"
            }

            st.subheader("Automatic Condition Prediction")

            col_a, col_b = st.columns(2)

            col_a.metric(
                "Predicted Condition",
                condition_display_names.get(
                    predicted_condition,
                    predicted_condition
                )
            )

            col_b.metric(
                "Classifier Confidence",
                f"{condition_confidence:.1%}"
            )

        st.caption(
            f"Applied Processing: **{processing_description}**"
        )


        # Image comparison
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("1. Original Image")

            st.image(
                original_rgb,
                use_container_width=True
            )

            st.subheader("Original Ensemble Detections")

            st.image(
                cv2.cvtColor(
                    original_annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.metric(
                "Final People Detected",
                len(original_detections)
            )

        with col2:

            st.subheader("2. Processed Image")

            st.image(
                cv2.cvtColor(
                    processed_bgr,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.subheader("Processed Ensemble Detections")

            st.image(
                cv2.cvtColor(
                    processed_annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.metric(
                "Final People Detected",
                len(processed_detections)
            )


        # Comparison
        st.subheader("Ensemble Detection Comparison")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Original Detections",
            len(original_detections)
        )

        c2.metric(
            "Processed Detections",
            len(processed_detections)
        )

        c3.metric(
            "Change in Count",
            len(processed_detections) - len(original_detections),
            delta=len(processed_detections) - len(original_detections)
        )


        # Individual model results
        st.subheader("Individual Model Predictions")

        model_names = ["VisDrone", "PDD", "COCO"]

        original_counts = {
            name: sum(
                1 for item in original_predictions
                if item["model"] == name
            )
            for name in model_names
        }

        processed_counts = {
            name: sum(
                1 for item in processed_predictions
                if item["model"] == name
            )
            for name in model_names
        }

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.write("**Original Image**")

            for name in model_names:
                st.write(
                    f"{name}: {original_counts[name]} predictions"
                )

        with result_col2:
            st.write("**Processed Image**")

            for name in model_names:
                st.write(
                    f"{name}: {processed_counts[name]} predictions"
                )


        # Agreement details
        st.subheader("Ensemble Agreement")

        for index, detection in enumerate(original_detections, start=1):

            st.write(
                f"Person {index}: "
                f"{detection['agreement']}/3 models agree | "
                f"Average confidence: {detection['confidence']:.3f}"
            )


        st.warning(
            "Important evaluation note: Ensemble detection counts are "
            "predictions, not ground-truth measurements. They do not "
            "directly establish Precision, Recall or F1-score. "
            "The ensemble must be evaluated against labelled test images "
            "before claiming an improvement."
        )


    except Exception as error:

        st.error(
            f"An error occurred while processing the image: {error}"
        )

else:

    st.info("Upload an aerial image to begin.")
