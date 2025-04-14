import streamlit as st
import numpy as np
from PIL import Image
import os

# Add error handling for imports
try:
    import cv2
    from ultralytics import YOLO
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = str(e)

class TumorDetectionApp:
    def __init__(self):
        try:
            model_path = 'exp4_best.pt'
            # Check if model exists
            if not os.path.exists(model_path):
                st.error(f"Model file {model_path} not found!")
                self.model = None
            else:
                self.model = YOLO(model_path)
            self.confidence_threshold = 0.5
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            self.model = None
    
    def process_image(self, image):
        if self.model is None:
            return np.array(image)  # Return original image if model not loaded
            
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            results = self.model(image)
            processed_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    if confidence > self.confidence_threshold:
                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                        label = f"{'Normal' if class_id == 0 else 'Tumor'} {confidence:.2f}"
                        cv2.putText(processed_image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return processed_image
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return image  # Return original image on error

def set_background():
    st.markdown("""
        <style>
            body {
                background-color: lavender;
            }
            .stApp {
                background-color: lavender;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Smart Scan - Tumor Detection", layout="wide")
    set_background()
    
    # Check for import errors first
    if IMPORT_ERROR:
        st.error(f"Error loading dependencies: {IMPORT_ERROR}")
        st.info("Please refresh the page or try again later. If the issue persists, contact support.")
        return
    
    try:
        detector = TumorDetectionApp()
        
        st.title("Smart Scan")
        st.write("Upload an X-ray image or use your webcam for real-time detection for Bone Tumor Detection")
        
        tab1, tab2 = st.tabs(["Image Upload", "Webcam Detection"])
        
        # --- Image Upload Tab ---
        with tab1:
            uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    if st.button("Detect Tumors"):
                        with st.spinner("Processing image..."):
                            processed_image = detector.process_image(image)
                            st.image(processed_image, caption="Detection Result", use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
        
        # --- Webcam Detection Tab ---
        with tab2:
            st.write("Position the X-ray in front of your webcam.")
            start_cam = st.checkbox("Start Real-time Detection")
            if start_cam:
                try:
                    webcam_image = st.camera_input("Take a picture")
                    if webcam_image is not None:
                        image = Image.open(webcam_image)
                        st.image(image, caption="Captured Image", use_container_width=True)
                        with st.spinner("Processing image..."):
                            processed_image = detector.process_image(image)
                            st.image(processed_image, caption="Detection Result", use_column_width=True)
                except Exception as e:
                    st.error(f"Error with webcam capture: {str(e)}")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == '__main__':
    main()
