
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PRETRAINED_MODEL_PATH = BASE_DIR / "yolo11n.pt"
TRAINED_MODEL_PATH = BASE_DIR / "models" / "visdrone_person_20epochs_best.pt"
CLASSIFIER_PATH = BASE_DIR / "models" / "condition_classifier_best.pt"

MODEL_PATH = (
    PRETRAINED_MODEL_PATH
    if PRETRAINED_MODEL_PATH.exists()
    else TRAINED_MODEL_PATH
)

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


def add_fog(image):
    white_layer = np.full_like(image, 255)
    return cv2.addWeighted(image, 0.55, white_layer, 0.45, 0)


def apply_clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def process_image(image, condition):
    if condition == "Clear (no enhancement)":
        return image.copy()

    if condition == "Synthetic fog":
        return add_fog(image)

    if condition == "CLAHE enhancement":
        return apply_clahe(image)

    if condition == "Fog + CLAHE (demonstration)":
        return apply_clahe(add_fog(image))

    return image.copy()


def detect_people(model, image, confidence):
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=1280,
        classes=[0],
        verbose=False
    )[0]

    annotated_image = result.plot()
    confidences = [
        float(box.conf[0].item())
        for box in result.boxes
    ]

    return annotated_image, len(result.boxes), confidences


st.title("🚁 RescueVision")
st.subheader("Disaster Image Enhancement and Person Detection")

st.write(
    "Upload an aerial image to predict its visual condition "
    "and compare person detections before and after processing."
)

if MODEL_PATH == PRETRAINED_MODEL_PATH:
    st.info("Person detector: pretrained YOLO11n (COCO).")
else:
    st.info("Person detector: fine-tuned VisDrone person model.")

if not CLASSIFIER_PATH.exists():
    st.error(
        "Condition classifier not found. Expected file: "
        "models/condition_classifier_best.pt"
    )
    st.stop()

st.caption(
    "The condition classifier is trained on a dataset containing "
    "synthetic weather examples. Its predictions require testing "
    "on realistic images. Dedicated deep-learning restoration "
    "models are not integrated yet."
)

st.sidebar.header("Detection Settings")

condition = st.sidebar.selectbox(
    "Processing demonstration",
    [
        "Clear (no enhancement)",
        "CLAHE enhancement",
        "Synthetic fog",
        "Fog + CLAHE (demonstration)"
    ]
)

confidence = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.25,
    step=0.01
)

st.sidebar.caption(
    "Higher thresholds usually show fewer, more confident detections."
)

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

        with st.spinner("Classifying condition and detecting people..."):
            classifier = load_model(str(CLASSIFIER_PATH))

            predicted_condition, condition_confidence = (
                classify_condition(classifier, original_bgr)
            )

            processed_bgr = process_image(original_bgr, condition)

            if not MODEL_PATH.exists():
                st.error(
                    "No person detector found. Put yolo11n.pt "
                    "in the project root or the fine-tuned model "
                    "inside the models folder."
                )
                st.stop()

            detector = load_model(str(MODEL_PATH))

            original_annotated, original_count, original_conf = (
                detect_people(detector, original_bgr, confidence)
            )

            processed_annotated, processed_count, processed_conf = (
                detect_people(detector, processed_bgr, confidence)
            )

        st.subheader("Automatic condition prediction")

        condition_names = {
            "clear": "Clear",
            "fog": "Fog",
            "rain": "Rain",
            "low_light": "Low light"
        }

        col_a, col_b = st.columns(2)

        col_a.metric(
            "Predicted condition",
            condition_names.get(
                predicted_condition,
                predicted_condition
            )
        )

        col_b.metric(
            "Classifier confidence",
            f"{condition_confidence:.1%}"
        )

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the condition is correct."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(original_rgb, use_container_width=True)

            st.subheader("Person detections — original")
            st.image(
                cv2.cvtColor(
                    original_annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.metric("People detected", original_count)
            st.write(
                "Confidence scores:",
                [round(x, 3) for x in original_conf]
                or "No detections"
            )

        with col2:
            st.subheader(f"Processed image: {condition}")
            st.image(
                cv2.cvtColor(
                    processed_bgr,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.subheader("Person detections — processed")
            st.image(
                cv2.cvtColor(
                    processed_annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.metric("People detected", processed_count)
            st.write(
                "Confidence scores:",
                [round(x, 3) for x in processed_conf]
                or "No detections"
            )

        st.subheader("Detection comparison")

        c1, c2, c3 = st.columns(3)
        c1.metric("Original detections", original_count)
        c2.metric("Processed detections", processed_count)
        c3.metric(
            "Change in count",
            processed_count - original_count
        )

        st.warning(
            "Detection counts alone cannot establish better accuracy. "
            "Ground-truth annotations and precision, recall, and "
            "F1-score are needed. Synthetic fog adds fog; it does "
            "not remove real fog."
        )

    except Exception as error:
        st.error(
            f"An error occurred while processing the image: {error}"
        )

else:
    st.info("Upload an aerial image to begin.")
