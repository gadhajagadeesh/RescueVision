import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PRETRAINED_MODEL_PATH = BASE_DIR / "yolo11n.pt"
TRAINED_MODEL_PATH = BASE_DIR / "models" / "visdrone_person_20epochs_best.pt"
MODEL_PATH = PRETRAINED_MODEL_PATH if PRETRAINED_MODEL_PATH.exists() else TRAINED_MODEL_PATH

st.set_page_config(page_title="RescueVision", page_icon="🚁", layout="wide")


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


def add_fog(image):
    white_layer = np.full_like(image, 255)
    return cv2.addWeighted(image, 0.55, white_layer, 0.45, 0)


def apply_clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def process_image(image, condition):
    if condition == "Clear (no enhancement)":
        return image.copy()
    if condition == "Synthetic fog":
        return add_fog(image)
    if condition == "Fog + CLAHE":
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
    confidences = [float(box.conf[0].item()) for box in result.boxes]
    return annotated_image, len(result.boxes), confidences


st.title("🚁 RescueVision")
st.subheader("Disaster Image Enhancement and Person Detection")
st.write(
    "Upload an aerial image, select a processing condition, and compare "
    "person detections on the original and processed images."
)

if MODEL_PATH == PRETRAINED_MODEL_PATH:
    st.info("Detector: pretrained YOLO11n (COCO), filtered to the person class.")
else:
    st.warning(
        "The pretrained yolo11n.pt file was not found. The app is using the "
        "fine-tuned VisDrone person model instead."
    )

st.caption(
    "Current prototype: manual condition selection, synthetic fog, CLAHE, "
    "and person detection. Automatic condition classification and rain/low-light "
    "enhancement are not implemented yet."
)

st.sidebar.header("Detection Settings")
condition = st.sidebar.selectbox(
    "Select image condition",
    ["Clear (no enhancement)", "Synthetic fog", "Fog + CLAHE"]
)
confidence = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.25,
    step=0.01
)
st.sidebar.caption("Higher thresholds usually show fewer, more confident detections.")

uploaded_file = st.file_uploader(
    "Upload a drone or aerial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        original_rgb = np.array(pil_image)
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        processed_bgr = process_image(original_bgr, condition)

        if not MODEL_PATH.exists():
            st.error(
                "No model file found. Put yolo11n.pt in the project root, or "
                "put visdrone_person_20epochs_best.pt inside the models folder."
            )
            st.stop()

        with st.spinner("Loading model and detecting people..."):
            model = load_model(str(MODEL_PATH))
            original_annotated, original_count, original_conf = detect_people(
                model, original_bgr, confidence
            )
            processed_annotated, processed_count, processed_conf = detect_people(
                model, processed_bgr, confidence
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original image")
            st.image(original_rgb, use_container_width=True)
            st.subheader("Person detections — original")
            st.image(
                cv2.cvtColor(original_annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            st.metric("People detected", original_count)
            st.write("Confidence scores:", [round(x, 3) for x in original_conf] or "No detections")

        with col2:
            st.subheader(f"Processed image: {condition}")
            st.image(
                cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            st.subheader("Person detections — processed")
            st.image(
                cv2.cvtColor(processed_annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
            st.metric("People detected", processed_count)
            st.write("Confidence scores:", [round(x, 3) for x in processed_conf] or "No detections")

        st.subheader("Detection comparison")
        c1, c2, c3 = st.columns(3)
        c1.metric("Original detections", original_count)
        c2.metric("Processed detections", processed_count)
        c3.metric("Change in count", processed_count - original_count)

        st.warning(
            "A higher detection count does not automatically mean better accuracy. "
            "Ground-truth labels and metrics such as precision, recall, and F1-score "
            "are needed to evaluate whether enhancement truly helps."
        )

    except Exception as error:
        st.error(f"An error occurred while processing the image: {error}")
else:
    st.info("Upload an aerial image to begin.")
