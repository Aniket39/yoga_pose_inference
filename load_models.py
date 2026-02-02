from ultralytics import YOLO
import joblib
import os

# Paths
POSE_MODEL_PATH = "models/yolov8n-pose.pt"
CLASSIFIER_PATH = "models/pose_geometry_classifier.pkl"

# Check files exist
assert os.path.exists(POSE_MODEL_PATH), "Pose model not found!"
assert os.path.exists(CLASSIFIER_PATH), "Classifier not found!"

# Load pose model
pose_model = YOLO(POSE_MODEL_PATH)
print("YOLOv8 Pose model loaded")

# Load classifier
pose_clf = joblib.load(CLASSIFIER_PATH)
print("Pose geometry classifier loaded")

