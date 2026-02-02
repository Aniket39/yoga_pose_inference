import torch
import ultralytics
import sklearn
import cv2
import joblib

print("Torch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)
print("Joblib OK")

print("CUDA available:", torch.cuda.is_available())
