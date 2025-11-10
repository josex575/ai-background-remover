import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import cv2

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker")
st.markdown("""
Upload a portrait photo, and the AI will:
- Remove the background  
- Center and crop the head with 2cm space above  
- Include neck and shoulders  
- Replace the background with plain white  
""")

uploaded = st.file_uploader("📸 Upload a front-facing portrait photo", type=["jpg", "jpeg", "png"])

# ---- FACE DETECTION FUNCTION ----
def detect_face(image: Image.Image):
    """Detects a face using OpenCV and returns bounding box (x, y, w, h)."""
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])  # largest face

# ---- CROP + ALIGN FUNCTION ----
def crop_passport_style(image: Image.Image, face_box):
    """Crops the image to include head, neck, and shoulders with 2cm space above head."""
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # Approximate pixels for 2cm above head (depends on resolution, assume 300 dpi)
    dpi = 300
    cm2px = dpi / 2.54  # 1 inch = 2.54 cm
    top_margin = int(2 * cm2px)

    # Expand crop box
    x1 = max(x - w // 4, 0)                  # wider left margin for shoulders
    x2 = min(x + w + w // 4, img_w)         # wider right margin for shoulders
    y1 = max(y - top_margin, 0)             # 2cm above head
    y2 = min(y + int(h * 2.5), img_h)       # extend downward to include neck/shoulders

    cropped = image.crop((x1, y1, x2, y2))

    # Resize to standard passport ratio (4:5)
    target_ratio = 4 / 5
    cropped_w, cropped_h = cropped.size
    current_ratio = cropped_w / cropped_h

    if current_ratio > target_ratio:
        # too wide → pad height
        new_h = int(cropped_w / target_ratio)
        pad_h = new_h - cropped_h
        cropped = ImageOps.expand(cropped, border=(0, pad_h // 2), fill="white")
    elif current_ratio < target_ratio:
        # too tall → pad width
        new_w = int(cropped_h * target_ratio)
        pad_w = new_w - cropped_w
        cropped = ImageOps.expand(cropped, border=(pad_w // 2, 0), fill="white")

    return cropped

# ---- BACKGROUND REPLACEMENT ----
def replace_background_with_white(image: Image.Image):
    """Removes background and replaces it with plain white."""
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

# ---- MAIN APP LOGIC ----
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)

    with st.spinner("🪄 Processing image..."):
        face_box = detect_face(image)
        if face_box is None:
            st.error("😕 No face detected. Please upload a clear front-facing portrait photo.")
        else:
            face_box = tuple(map(int, face_box))
            cropped = crop_passport_style(image, face_box)
            final = replace_background_with_white(cropped)
            final = ImageOps.autocontrast(final)

            st.image(final, caption="✅ Passport-Ready Photo", use_container_width=True)

            # Download button
            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            st.download_button(
                label="💾 Download Passport Photo",
                data=byte_im,
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )
else:
    st.info("👆 Upload a clear, front-facing portrait photo.")
