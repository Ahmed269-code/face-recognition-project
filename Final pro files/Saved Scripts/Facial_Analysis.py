import streamlit as st
from deepface import DeepFace
import cv2
import tempfile
import os
from datetime import datetime
from PIL import Image
import numpy as np

# --- Constants ---
TEMP_IMAGE_PATH = "temp_analyze.jpg"
MAX_DISPLAY_WIDTH = 400  # Maximum width for displayed images

# --- Image Resizing Function ---
def resize_image(image, max_width=MAX_DISPLAY_WIDTH):
    """Resize image to a reasonable display size while maintaining aspect ratio"""
    if isinstance(image, np.ndarray):  # OpenCV image
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            resized = cv2.resize(image, (max_width, new_height))
            return resized, (width, height)
        return image, (width, height)
    else:  # PIL image
        width, height = image.size
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            resized = image.resize((max_width, new_height))
            return resized, (width, height)
        return image, (width, height)

# --- Face Analysis ---
def analyze_faces(frame):
    try:
        # Analyze facial attributes
        results = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'emotion', 'race'],
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True
        )
        
        # Process results (handle single/multiple faces)
        if isinstance(results, list):
            return results  # Multiple faces
        else:
            return [results]  # Single face as list
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return []

# --- Streamlit UI ---
def main():
    st.markdown("Analyze age, gender, emotion, and race using DeepFace")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a photo containing faces"
    )
    
    if uploaded_file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name
        
        try:
            # Read and process image
            img = cv2.imread(input_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Analyze faces
            with st.spinner("Analyzing facial attributes..."):
                analysis_results = analyze_faces(input_path)
            
            if analysis_results:
                # Display results
                st.success("Analysis complete!")
                
                # Resize original image for display
                resized_img, original_size = resize_image(img_rgb)
                
                # Create columns for image and attributes
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(resized_img, caption=f"Uploaded Image (Original: {original_size[0]}×{original_size[1]})")
                
                with col2:
                    st.subheader("Detected Faces")
                    for i, face in enumerate(analysis_results, 1):
                        with st.container():
                            st.markdown(f"**Face {i} Attributes**")
                            st.write(f"**Age**: {face['age']} years")
                            st.write(f"**Gender**: {face['dominant_gender']}")
                            st.write(f"**Emotion**: {face['dominant_emotion']}")
                            st.write(f"**Race**: {face['dominant_race']}")
                            st.write("")  # Spacer
                
                # Detailed view in expanders
                st.subheader("Detailed Analysis")
                for i, face in enumerate(analysis_results, 1):
                    with st.expander(f"📊 Detailed Analysis for Face {i}", expanded=False):
                        # Emotion distribution chart
                        st.subheader("Emotion Distribution")
                        emotions = face['emotion']
                        st.bar_chart(emotions)
                        
                        # All attributes in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Age", f"{face['age']} years")
                            st.metric("Gender", face['dominant_gender'])
                        with col2:
                            st.metric("Emotion", face['dominant_emotion'])
                            st.metric("Race", face['dominant_race'])
                
                # Download button
                output_path = f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Original Image",
                        f,
                        output_path,
                        "image/jpeg"
                    )
                os.unlink(output_path)
            else:
                st.warning("No faces detected or analysis failed")
        
        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.unlink(input_path)