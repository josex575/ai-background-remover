import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import cv2

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker")
st.markdown("""
Upload a photo, and the AI will:
- Remove the background  
- Center and crop the face (70–80% of frame)  
- Replace the background with plain white  
""")

uploaded = st.file_uploader("📸 Upload a clear portrait photo", type=["jpg", "jpeg", "png"])

# ---- FACE DETECTION FUNCTION ----
def detect_face(image: Image.Image):
    """Detects a face using OpenCV and returns bounding box (x, y, w, h)."""
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    # Return the largest detected face
    return max(faces, key=lambda rect: rect[2] * rec*
