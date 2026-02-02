import streamlit as st
import cv2
import numpy as np
import joblib
import time
import math
from ultralytics import YOLO
from PIL import Image

# --------------------------------
# Page config (STEP 7 – UI polish)
# --------------------------------
st.set_page_config(
    page_title="Yoga Pose Prediction",
    layout="centered"
)

st.title("Yoga Pose Prediction")
st.write("Upload a yoga image to detect pose using **YOLOv8 Pose + Geometry Classification**")

# --------------------------------
# Load models (cached)
# --------------------------------
@st.cache_resource
def load_models():
    pose_model = YOLO("models/yolov8n-pose.pt")
    clf = joblib.load("models/pose_geometry_classifier.pkl")
    class_names = joblib.load("models/class_names.pkl")
    return pose_model, clf, class_names

pose_model, clf, class_names = load_models()

# --------------------------------
# STEP 3 – Geometry + angle helpers
# --------------------------------
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_pose_features(kp):
    feats = []

    # Angles (4)
    feats.append(compute_angle(kp[5], kp[7], kp[9]))      # left arm
    feats.append(compute_angle(kp[6], kp[8], kp[10]))    # right arm
    feats.append(compute_angle(kp[11], kp[13], kp[15]))  # left leg
    feats.append(compute_angle(kp[12], kp[14], kp[16]))  # right leg

    # Distances (4)
    feats.append(np.linalg.norm(kp[5] - kp[7]))
    feats.append(np.linalg.norm(kp[7] - kp[9]))
    feats.append(np.linalg.norm(kp[11] - kp[13]))
    feats.append(np.linalg.norm(kp[13] - kp[15]))

    return np.array(feats).reshape(1, -1)

# --------------------------------
# Skeleton definition
# --------------------------------
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (11, 12),
    (5, 11), (6, 12)
]

# --------------------------------
# Image upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload a yoga image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # --------------------------------
    # STEP 5 – Inference time
    # --------------------------------
    start_time = time.time()
    results = pose_model(img_bgr, conf=0.25, iou=0.5, max_det=1, verbose=False)
    inference_time = time.time() - start_time

    if results[0].keypoints is None:
        st.error("❌ No person detected in the image")
    else:
        kp = results[0].keypoints.xy[0].cpu().numpy()

        # Draw keypoints
        for x, y in kp:
            cv2.circle(img_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Draw skeleton
        for i, j in SKELETON:
            x1, y1 = kp[i]
            x2, y2 = kp[j]
            cv2.line(
                img_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )

        # --------------------------------
        # STEP 4 – Pose classification
        # --------------------------------
        features = extract_pose_features(kp)
        pred_idx = clf.predict(features)[0]
        pred_name = class_names[pred_idx]

        # Convert back to RGB
        output_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --------------------------------
        # STEP 6 – Save output
        # --------------------------------
        output_path = "outputs/predicted_pose.png"
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

        # --------------------------------
        # STEP 7 – Final UI
        # --------------------------------
        st.subheader("Pose Detection Output")
        st.image(output_img, use_container_width=True)

        st.success(f"🧘 Predicted Pose: **{pred_name}**")
        st.info(f"⏱️ Inference Time: **{inference_time:.3f} sec**")

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Output Image",
                data=f,
                file_name="predicted_pose.png",
                mime="image/png"
            )
