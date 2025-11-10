import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import cv2

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker")
st.markdown("""
Upload a portrait photo and select the type below.  
The AI will automatically crop, center, and clean up the background for a passport-ready image.
""")

# ---- USER OPTIONS ----
photo_type = st.selectbox(
    "🧔 Photo Type",
    ["Without Beard", "With Beard"]
)

subject_type = st.selectbox(
    "👤 Subject Type",
    ["Man", "Woman", "Baby"]
)

uploaded = st.file_uploader("📸 Upload a clear, front-facing portrait photo", type=["jpg", "jpeg", "png"])

# ---- FACE DETECTION ----
def detect_face(image):
    """Detects the largest face and returns (x, y, w, h)."""
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])

# ---- CROP FUNCTION ----
def crop_based_on_type(image, face_box, photo_type, subject_type):
    """
    Crops image with different margins based on subject and beard selection.
    """
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # cm → pixels (300 DPI)
    dpi = 300
    cm2px = dpi / 2.54
    top_margin = int(2 * cm2px)

    # Base bottom margin depending on subject type
    if subject_type == "Man":
        bottom_margin = int(h * 0.12)
    elif subject_type == "Woman":
        bottom_margin = int(h * 0.15)
    else:  # Baby
        bottom_margin = int(h * 0.25)  # babies have rounder faces and need more bottom space

    # Add extra 2 cm below beard if selected
    if photo_type == "With Beard":
        bottom_margin += int(2 * cm2px)

    # Horizontal margins (slightly wider than face)
    x1 = max(x - w // 8, 0)
    x2 = min(x + w + w // 8, img_w)

    # Vertical margins
    y1 = max(y - top_margin, 0)
    y2 = min(y + h + bottom_margin, img_h)

    cropped = image.crop((x1, y1, x2, y2))

    # Adjust to 4:5 ratio
    target_ratio = 4 / 5
    cw, ch = cropped.size
    ratio = cw / ch

    if ratio > target_ratio:
        new_h = int(cw / target_ratio)
        pad_h = new_h - ch
        cropped = ImageOps.expand(cropped, border=(0, pad_h // 2), fill="white")
    elif ratio < target_ratio:
        new_w = int(ch * target_ratio)
        pad_w = new_w - cw
        cropped = ImageOps.expand(cropped, border=(pad_w // 2, 0), fill="white")

    return cropped

# ---- BACKGROUND REMOVAL ----
def replace_background_with_white(image):
    """Removes background and replaces it with white."""
    removed = remove(image)
    np_img = np.array(removed)
    if np_img.shape[2] == 4:
        alpha = np_img[:, :, 3]
        white_bg = np.ones_like(np_img[:, :, :3]) * 255
        alpha_factor = alpha[:, :, np.newaxis] / 255.0
        composite = white_bg * (1 - alpha_factor) + np_img[:, :, :3] * alpha_factor
        return Image.fromarray(composite.astype(np.uint8))
    else:
        return image

# ---- MAIN APP ----
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)

    with st.spinner("🪄 Processing photo..."):
        face_box = detect_face(image)
        if face_box is None:
            st.error("😕 No face detected. Please upload a clear, front-facing portrait photo.")
        else:
            face_box = tuple(map(int, face_box))

            cropped = crop_based_on_type(image, face_box, photo_type, subject_type)
            final = replace_background_with_white(cropped)
            final = ImageOps.autocontrast(final)

            # Resize final output to 630x810
            final = final.resize((630, 810), Image.LANCZOS)

            st.image(final, caption=f"✅ {subject_type} ({photo_type}) Passport Photo (630×810 px)", use_container_width=True)

            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download Passport Photo",
                data=buf.getvalue(),
                file_name=f"{subject_type.lower()}_{photo_type.lower().replace(' ', '_')}_passport_photo.jpg",
                mime="image/jpeg"
            )
else:
    st.info("👆 Upload a clear, front-facing portrait photo.")
