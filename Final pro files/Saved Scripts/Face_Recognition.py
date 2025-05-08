import streamlit as st
import cv2
import numpy as np
import joblib
import json
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tempfile
import os
from datetime import datetime
import pandas as pd
from collections import deque
from PIL import Image
import time  # Added for time measurement

# --- Constants ---
RECOGNITION_INTERVAL = 1  # Process 1 frame per second
MIN_CONFIDENCE = 30  # Minimum confidence to count as recognized
TEMP_VIDEO_PATH = "processed_output.mp4"
TEMP_IMAGE_PATH = "processed_image.jpg"
BBOX_COLORS = {
    "VERIFIED": (0, 255, 0),    # Green
    "UNKNOWN": (255, 0, 0)      # Red
}

# --- Face Tracker Class ---
class FaceTracker:
    def __init__(self, max_age=10):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        
    def update(self, detections):
        # Age all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # Match detections to tracks
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track['age'] == 0:  # Already matched
                continue
                
            best_match = None
            best_iou = 0.3  # Minimum overlap threshold
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                    
                iou = self._calculate_iou(track['bbox'], det['facial_area'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                # Update track
                self.tracks[track_id]['bbox'] = detections[best_match]['facial_area']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['data'] = detections[best_match]
                matched_detections.add(best_match)
        
        # Add new detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': det['facial_area'],
                    'age': 0,
                    'data': det
                }
                self.next_id += 1
                
        return self.tracks
    
    def _calculate_iou(self, box1, box2):
        # Calculate Intersection over Union
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x']+box1['w'], box2['x']+box2['w'])
        y2 = min(box1['y']+box1['h'], box2['y']+box2['h'])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1['w'] * box1['h']
        box2_area = box2['w'] * box2['h']
        
        return inter_area / float(box1_area + box2_area - inter_area)

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("svm_model_1.joblib", 'rb') as f:
        svm = joblib.load(f)
    with open("label_mapping.json") as f:
        label_mapping = json.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return svm, {str(v):k for k,v in label_mapping.items()}, embeddings

# --- Enhanced Face Visualization ---
def draw_face_annotations(frame, recognition):
    x, y, w, h = recognition['facial_area']['x'], recognition['facial_area']['y'], \
                 recognition['facial_area']['w'], recognition['facial_area']['h']
    
    color = BBOX_COLORS.get(recognition['status'], (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw text with background
    text = recognition.get('name', 'Unknown') if recognition['status'] == "VERIFIED" else "UNKNOWN"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y-30), (x + text_width + 10, y), color, -1)
    cv2.putText(frame, text, (x+5, y-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

# --- Face Recognition ---
def recognize_faces(frame, svm, inv_map, embeddings):
    results = []
    try:
        detected_faces = DeepFace.represent(
            img_path=frame,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=False
        )
        
        for face in detected_faces:
            if isinstance(face, dict) and "embedding" in face:
                embedding = np.array(face["embedding"]).reshape(1, -1)
                
                # SVM prediction
                svm_label = svm.predict(embedding)[0]
                svm_name = inv_map.get(str(svm_label), "Unknown")
                svm_proba = svm.predict_proba(embedding)[0][svm_label]
                
                # Cosine similarity
                all_embeddings = np.concatenate(list(embeddings.values()))
                similarities = cosine_similarity(embedding, all_embeddings)[0]
                best_match_score = np.max(similarities)
                
                # Calculate confidence and determine status
                confidence = (svm_proba + best_match_score) * 50
                status = "VERIFIED" if confidence >= MIN_CONFIDENCE else "UNKNOWN"
                
                # Set name to "Unknown person" if status is UNKNOWN
                display_name = svm_name if status == "VERIFIED" else "Unknown person"
                
                results.append({
                    "name": display_name,
                    "confidence": confidence,
                    "facial_area": face["facial_area"],
                    "status": status,
                    "embedding": embedding
                })
    except Exception as e:
        st.warning(f"Recognition skipped: {str(e)}")
    return results

# --- Image Processing ---
def process_image(image_path, output_path, svm, inv_map, embeddings):
    start_time = time.time()  # Start timing
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    recognitions = recognize_faces(img_rgb, svm, inv_map, embeddings)
    attendance = {}
    all_records = []
    
    for rec in recognitions:
        # Update attendance
        if rec['status'] == "VERIFIED":
            name = rec['name']
            if name not in attendance:
                attendance[name] = {
                    'count': 1
                }
            else:
                attendance[name]['count'] += 1
        
        # Draw annotations
        draw_face_annotations(img_rgb, rec)
        
        # Record for CSV (only name and status)
        all_records.append({
            "name": rec['name'],
            "status": rec['status']
        })
    
    # Save processed image
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    elapsed = time.time() - start_time  # Calculate elapsed time
    st.info(f"Image processing time: {elapsed:.2f} seconds")
    
    return attendance, pd.DataFrame(all_records), img_rgb

# --- Video Processing with Tracking ---
def process_video(input_path, output_path, svm, inv_map, embeddings):
    start_time = time.time()  # Start timing
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = FaceTracker(max_age=int(fps))  # Keep tracks for 1 second
    attendance = {}
    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_records = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process at intervals
        if frame_count % int(fps/RECOGNITION_INTERVAL) == 0:
            recognitions = recognize_faces(frame_rgb, svm, inv_map, embeddings)
            active_tracks = tracker.update(recognitions)
        else:
            active_tracks = tracker.tracks
        
        # Update attendance and draw boxes
        for track_id, track in active_tracks.items():
            if track['age'] == 0:  # Only fresh detections
                rec = track['data']
                
                # Update attendance
                if rec['status'] == "VERIFIED":
                    name = rec['name']
                    if name not in attendance:
                        attendance[name] = {
                            'first_seen': frame_count/fps,
                            'last_seen': frame_count/fps,
                            'count': 1
                        }
                    else:
                        attendance[name]['last_seen'] = frame_count/fps
                        attendance[name]['count'] += 1
                
                # Draw annotations
                draw_face_annotations(frame_rgb, rec)
                
                # Record for CSV (only name and status)
                all_records.append({
                    "frame": frame_count,
                    "time": frame_count/fps,
                    "name": rec['name'],
                    "status": rec['status']
                })
        
        out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Update progress
        if frame_count % 10 == 0:
            progress = min(frame_count/(fps*60), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    elapsed = time.time() - start_time  # Calculate elapsed time
    st.info(f"Video processing time: {elapsed:.2f} seconds ({frame_count} frames at {fps:.1f} FPS)")
    
    return attendance, pd.DataFrame(all_records)

# --- Streamlit UI ---
def main():    
    # Load models
    svm, inv_map, embeddings = load_models()    
    # File upload
    file_type = st.radio("Select input type:", ("Image", "Video"))
    
    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name
        
        # Process based on file type
        if file_type == "Image":
            attendance, all_records, processed_img = process_image(
                input_path, TEMP_IMAGE_PATH,
                svm, inv_map, embeddings
            )
            
            # Display results
            st.success("Processing complete!")
            
            # Show processed image
            st.header("Annotated Image")
            st.image(processed_img, channels="RGB", width=500)
            
            # Download buttons
            st.header("Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                with open(TEMP_IMAGE_PATH, "rb") as f:
                    st.download_button(
                        "Download Processed Image",
                        f,
                        f"processed_image_{datetime.now().strftime('%Y%m%d')}.jpg",
                        "image/jpeg"
                    )
            
            with col2:
                if not attendance:
                    st.warning("No faces detected")
                else:
                    # Simplified CSV with just names and counts
                    df = pd.DataFrame({'name': attendance.keys(), 'count': [v['count'] for v in attendance.values()]})
                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        "Download Recognition Report (CSV)",
                        csv,
                        f"recognition_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
            
            # Show detailed records
            if not all_records.empty:
                st.header("Detailed Recognition Log")
                st.dataframe(all_records[['name', 'status']])
            
            # Cleanup
            os.unlink(input_path)
            if os.path.exists(TEMP_IMAGE_PATH):
                os.unlink(TEMP_IMAGE_PATH)
                
        else:  # Video processing
            st.header("Video Processing")
            attendance, all_records = process_video(
                input_path, TEMP_VIDEO_PATH,
                svm, inv_map, embeddings
            )
            
            # Display results
            st.success("Processing complete!")
            
            # Show processed video
            st.header("Annotated Video")
            st.video(TEMP_VIDEO_PATH)
            
            # Download buttons
            st.header("Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                with open(TEMP_VIDEO_PATH, "rb") as f:
                    st.download_button(
                        "Download Processed Video",
                        f,
                        f"attendance_video_{datetime.now().strftime('%Y%m%d')}.mp4",
                        "video/mp4"
                    )
            
            with col2:
                if not attendance:
                    st.warning("No attendance data to export")
                else:
                    # Simplified CSV with just names, first_seen, last_seen, and count
                    df = pd.DataFrame({
                        'name': attendance.keys(),
                        'first_seen': [v['first_seen'] for v in attendance.values()],
                        'last_seen': [v['last_seen'] for v in attendance.values()],
                        'count': [v['count'] for v in attendance.values()]
                    })
                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        "Download Attendance Report (CSV)",
                        csv,
                        f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
            
            # Show detailed records
            if not all_records.empty:
                st.header("Detailed Recognition Log")
                st.dataframe(all_records[['frame', 'time', 'name', 'status']])
            
            # Cleanup
            os.unlink(input_path)
            if os.path.exists(TEMP_VIDEO_PATH):
                os.unlink(TEMP_VIDEO_PATH)