import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from datetime import datetime

st.set_page_config(page_title="SmartVision - Object Detection", page_icon="🎯", layout="wide")
st.title("🎯 SmartVision: Real-Time Object Detection using OpenCV + Streamlit")
st.markdown("Detect **Faces**, **Eyes**, **Full Body**, or **Cars** from Images, Videos, or Webcam.")


os.makedirs("output/images", exist_ok=True)
os.makedirs("output/videos", exist_ok=True)


st.sidebar.header("⚙️ Detection Settings")
detection_type = st.sidebar.selectbox(
    "Choose what to detect",
    ("Face", "Eyes", "Full Body", "Cars")
)

#  Haar Cascade path
cascade_dict = {
    "Face": "haarcascades/haarcascade_frontalface_default.xml",
    "Eyes": "haarcascades/haarcascade_eye.xml",
    "Full Body": "haarcascades/haarcascade_fullbody.xml",
    "Cars": "haarcascades/haarcascade_car.xml"
}

cascade_path = cascade_dict[detection_type]

# Load Haar Cascade
if not os.path.exists(cascade_path):
    st.error(f"❌ Haar Cascade not found at {cascade_path}. Please check your folder structure.")
    st.stop()

detector = cv2.CascadeClassifier(cascade_path)


# Input Type

option = st.radio("Select Input Type", ("Image", "Video", "Live Webcam"), horizontal=True)


# IMAGE MODE

if option == "Image":
    uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        objects = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in objects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)

        if st.button("💾 Save Image"):
            filename = f"output/images/{detection_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, image)
            st.success(f"✅ Saved image to `{filename}`")


# VIDEO MODE

elif option == "Video":
    uploaded_video = st.file_uploader("🎥 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out_path = f"output/videos/{detection_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

        stframe = st.empty()
        st.info("⏳ Processing video... please wait.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            out.write(frame)

        cap.release()
        out.release()
        st.success(f"✅ Video saved at `{out_path}`")


# LIVE WEBCAM MODE

elif option == "Live Webcam":
    st.info("Click **Start Webcam** to begin detection.")
    start_button = st.button("▶️ Start Webcam")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_button = st.button("⏹ Stop Webcam")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()
        st.success("✅ Webcam stopped.")