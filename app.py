import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch


class TumorDetectionApp:
    def __init__(self):
        self.model = YOLO('exp4_best.pt')  # Load your trained YOLO model
        self.confidence_threshold = 0.5
       
    def process_image(self, image):
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
           
        # Make prediction
        results = self.model(image)
       
        # Process results
        processed_image = image.copy()
       
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
               
                # Get confidence and class
                confidence = float(box.conf)
                class_id = int(box.cls)
               
                if confidence > self.confidence_threshold:
                    # Draw bounding box
                    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for normal, Red for tumor
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                   
                    # Add label
                    label = f"{'Normal' if class_id == 0 else 'Tumor'} {confidence:.2f}"
                    cv2.putText(processed_image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
        return processed_image


def main():
    st.set_page_config(page_title="X-Ray Tumor Detection", layout="wide")
   
    # Initialize the app
    detector = TumorDetectionApp()
   
    # Add title and description
    st.title("Smart Scan")
    st.write("Upload an X-ray image or use your webcam for real-time detection for Bone Tumor Detection")
   
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Image Upload", "Webcam Detection"])
   
    # Image Upload Tab
    with tab1:
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])
       
        if uploaded_file is not None:
            # Read and process the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
           
            if st.button("Detect Tumors"):
                processed_image = detector.process_image(image)
                st.image(processed_image, caption="Detection Result", use_column_width=True)
   
    # Webcam Detection Tab
    with tab2:
        st.write("Position the X-ray in front of your webcam")
        run = st.checkbox("Start Real-time Detection")
       
        if run:
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0)
           
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
               
                # Process frame
                processed_frame = detector.process_image(frame)
               
                # Convert BGR to RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
               
                # Update the frame
                FRAME_WINDOW.image(processed_frame)
           
            cap.release()


if __name__ == '__main__':
    main()

