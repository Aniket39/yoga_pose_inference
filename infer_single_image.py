import cv2
import numpy as np
import joblib
import time
from ultralytics import YOLO

# -----------------------
# Load models
# -----------------------
class_names = joblib.load("models/class_names.pkl")
pose_model = YOLO("models/yolov8n-pose.pt")
clf = joblib.load("models/pose_geometry_classifier.pkl")

# -----------------------
# Geometry helpers
# -----------------------
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_pose_features(kp):
    feats = []

    # ---- Angles (4) ----
    feats.append(compute_angle(kp[5], kp[7], kp[9]))     # left arm
    feats.append(compute_angle(kp[6], kp[8], kp[10]))   # right arm
    feats.append(compute_angle(kp[11], kp[13], kp[15])) # left leg
    feats.append(compute_angle(kp[12], kp[14], kp[16])) # right leg

    # ---- Distances (4) ----
    feats.append(np.linalg.norm(kp[5] - kp[7]))     # left upper arm
    feats.append(np.linalg.norm(kp[7] - kp[9]))     # left forearm
    feats.append(np.linalg.norm(kp[11] - kp[13]))   # left thigh
    feats.append(np.linalg.norm(kp[13] - kp[15]))   # left calf

    return np.array(feats).reshape(1, -1)

# -----------------------
# Skeleton connections (COCO)
# -----------------------
SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]

# -----------------------
# Load image
# -----------------------
IMAGE_PATH = "test_images/17.png"
img = cv2.imread(IMAGE_PATH)
assert img is not None, "❌ Image not found!"

# -----------------------
# ⏱️ START INFERENCE TIMER
# -----------------------
start_time = time.time()

# -----------------------
# Pose inference
# -----------------------
results = pose_model(
    img,
    conf=0.25,
    iou=0.5,
    max_det=1,
    verbose=False
)

if results[0].keypoints is None:
    print("❌ No person detected")
    exit()

kp = results[0].keypoints.xy[0].cpu().numpy()

# -----------------------
# Feature extraction + classification
# -----------------------
features = extract_pose_features(kp)
pred_idx = clf.predict(features)[0]
pred_name = class_names[pred_idx]

# -----------------------
# ⏱️ STOP TIMER
# -----------------------
end_time = time.time()
inference_time = end_time - start_time

# -----------------------
# Draw keypoints
# -----------------------
for x, y in kp:
    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

# -----------------------
# Draw skeleton
# -----------------------
for i, j in SKELETON:
    x1, y1 = kp[i]
    x2, y2 = kp[j]
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255, 0, 0),
        2
    )

# -----------------------
# Draw text (pose + time)
# -----------------------
cv2.putText(
    img,
    f"Predicted Pose: {pred_name}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.putText(
    img,
    f"Inference Time: {inference_time:.3f}s",
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)

# -----------------------
# Save output
# -----------------------
OUTPUT_PATH = "outputs/predicted_pose.png"
cv2.imwrite(OUTPUT_PATH, img)
print(f"✅ Output saved at: {OUTPUT_PATH}")
print(f"⏱️ Inference time: {inference_time:.3f} seconds")

# -----------------------
# Show image
# -----------------------
cv2.imshow("Yoga Pose Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
