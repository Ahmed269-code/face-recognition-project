import streamlit as st
from Face_Recognition import main as face_recognition_main
from Facial_Analysis import main as facial_analysis_main

# --- Set page config ---
st.set_page_config(
    page_title="Face Analysis Suite",
    page_icon="👥",
    layout="wide"
)

# --- Custom CSS to shrink sidebar ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            font-size: 14px;
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", [
    "Face Recognition",
    "Face Analysis"
])

# --- Main logic ---
if page == "Face Recognition":
    st.title("👥 Face Recognition System")
    face_recognition_main()

elif page == "Face Analysis":
    st.title("🧠 Facial Attribute Analysis")
    facial_analysis_main()