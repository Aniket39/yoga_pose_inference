# 🧘 Yoga Pose Classification using YOLOv8 & Geometry

This project classifies yoga poses from images using **human pose estimation** and **geometry-based machine learning**, instead of traditional image classification.

---

## 🚀 Features
- Human pose detection using YOLOv8-Pose
- Joint angle & distance feature extraction
- Geometry-based pose classification
- Real-time inference
- Streamlit web application
- Downloadable output image

---

## 🧠 Approach

Traditional image classification struggles with yoga poses due to background, lighting, and clothing variations.

This project solves that by:
1. Detecting body keypoints
2. Extracting angles and distances
3. Classifying poses using geometry

---

## 🏗️ Architecture

Input Image
→ YOLOv8 Pose Detection
→ Keypoints (17 joints)
→ Geometry + Angle Features
→ ML Classifier
→ Yoga Pose Prediction


---

## 📂 Project Structure

yoga_pose_inference/
│
├── app/
│ └── streamlit_app.py
│
├── models/
│ ├── yolov8n-pose.pt
│ ├── pose_geometry_classifier.pkl
│ └── class_names.pkl
│
├── test_images/
│ └── sample.png
│
├── outputs/
│ └── predicted_pose.png
│
├── infer_single_image.py
├── requirements.txt
└── README.md


---

## 🧪 Features Used for Classification

### Angles
- Left arm
- Right arm
- Left leg
- Right leg

### Distances
- Upper arm
- Forearm
- Thigh
- Calf

---

## 🖥️ Installation

```bash
git clone https://github.com/your-username/yoga-pose-classification.git
cd yoga-pose-classification
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
▶️ Run Streamlit App
streamlit run app/streamlit_app.py
📊 Sample Output
Predicted yoga pose name

Skeleton overlay

Inference time

Downloadable output image

⚠️ Notes
Accuracy depends on dataset quality

Some poses share similar geometry

Model predicts closest matching pose

📌 Future Improvements
Add pose correctness scoring

Webcam live inference

Normalize features

Increase dataset size

🙌 Acknowledgements
Ultralytics YOLOv8

OpenCV

Scikit-learn

Streamlit