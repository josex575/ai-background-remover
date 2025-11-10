import streamlit as st
from rembg import remove
from PIL import Image, ImageOps, ImageDraw
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
    """Detects a face using OpenCV and ret
