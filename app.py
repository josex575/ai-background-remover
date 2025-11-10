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
    return max(faces, key=lambda rect: rect[2] * rect[3])

# ---- CROP + ALIGN FUNCTION ----
def crop_passport_style(image: Image.Image, face_box):
    """Crops the image so that the head occupies ~75% of frame height."""
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # Compute desired crop size
    head_height_target = h / 0.75
    crop_h = int(head_height_target)
    crop_w = int(crop_h * 0.8)  # typical passport aspect ratio

    # Center the crop around the face
    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(center_x - crop_w // 2, 0)
    y1 = max(center_y - crop_h // 2, 0)
    x2 = min(x1 + crop_w, img_w)
    y2 = min(y1 + crop_h, img_h)

    cropped = image.crop((x1, y1, x2, y2))
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

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)

    with st.spinner("🪄 Processing image..."):
        face_box = detect_face(image)
        if not face_box:
            st.error("😕 No face detected. Try uploading a clearer portrait photo.")
        else:
            cropped = crop_passport_style(image, face_box)
            final = replace_background_with_white(cropped)
            final = ImageOps.autocontrast(final)

            st.image(final, caption="✅ Passport-Ready Photo", use_container_width=True)

            # Download
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
