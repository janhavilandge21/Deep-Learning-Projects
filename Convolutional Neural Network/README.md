# SmartVision-Real-Time-Object-Detection-App
I’m excited to share my latest computer vision project — SmartVision, a powerful Streamlit-based application that performs real-time object detection for faces, eyes, pedestrians, and cars using Haar Cascade classifiers.

# 🚀 Project Overview
The goal of SmartVision is to provide an easy-to-use web interface for object detection tasks — whether you’re analyzing a single image, processing a video, or using your live webcam.

This project demonstrates how OpenCV’s classical computer vision techniques can be combined with modern interactive frameworks like Streamlit to create an intuitive and real-time detection experience.

# 🔍 Features
✅ Detect Faces, Eyes, Full Body (Pedestrians), and Cars
✅ Choose input type: Image, Video, or Live Webcam
✅ Save processed results automatically into organized folders
✅ Real-time frame rendering inside Streamlit interface
✅ Lightweight and fast — no heavy deep learning models required
✅ Simple yet smart UI powered by Streamlit

# 🧩 Tech Stack
Python 3.10+
OpenCV (cv2)
Streamlit
NumPy
Haar Cascade Classifiers

# 🧠 How It Works
The app loads a selected Haar Cascade XML model for your chosen detection type.
It processes your uploaded image, video, or webcam feed frame by frame.
Detected objects are highlighted using bounding boxes in real time.
The user can download and save the output to a local directory.

# 🖥️ Running the App
Clone the repository and install dependencies:

pip install streamlit opencv-python numpy
# 💡 What I Learned
Practical use of OpenCV Haar Cascades for classical object detection.

Integrating computer vision models with Streamlit front-end.

Managing real-time video streaming efficiently within a web app.

Creating a clean and modular folder structure for CV applications

# 🌟 Future Enhancements
Add support for YOLOv8 / SSD deep learning-based detection.

Display object count statistics per frame.

Enable auto-download of processed videos.

Add dark mode UI with modern styling.
