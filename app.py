import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

class TumorDetectionApp:
    def __init__(self):
        self.model = YOLO('exp4_best.pt')  # Load your trained YOLO model
        self.confidence_threshold = 0.5

    def process_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = self.model(image)
        processed_image = image.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = float(box.conf)
                class_id = int(box.cls)

                if confidence > self.confidence_threshold:
                    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{'Normal' if class_id == 0 else 'Tumor'} {confidence:.2f}"
                    cv2.putText(processed_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return processed_image

def main():
    st.set_page_config(page_title="X-Ray Tumor Detection", layout="wide")

    detector = TumorDetectionApp()

    st.title("Smart Scan")
    st.write("Upload an X-ray image or use your webcam for real-time detection for Bone Tumor Detection")

    tab1, tab2 = st.tabs(["Image Upload", "Webcam Detection"])

    # --- Image Upload Tab ---
    with tab1:
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect Tumors"):
                processed_image = detector.process_image(image)
                st.image(processed_image, caption="Detection Result", use_column_width=True)

    # --- Webcam Detection Tab ---
    with tab2:
        st.write("Position the X-ray in front of your webcam and capture an image")

        webcam_image = st.camera_input("Take a picture")

        if webcam_image is not None:
            image = Image.open(webcam_image)
            st.image(image, caption="Captured Image", use_container_width=True)

            processed_image = detector.process_image(image)
            st.image(processed_image, caption="Detection Result", use_column_width=True)

if __name__ == '__main__':
    main()
