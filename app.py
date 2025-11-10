import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import cv2

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker")
st.markdown("""
Upload a portrait photo. The AI will:
- Remove background  
- Crop around the head with a small portion of neck  
- Leave **2 cm above the head**  
- Detect beards — if present, leave **2 cm extra below the beard**  
- Resize to **630×810 px**  
- Replace the background with white  
""")

uploaded = st.file_uploader("📸 Upload a clear front-facing portrait photo", type=["jpg", "jpeg", "png"])

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

# ---- BEARD DETECTION ----
def has_beard(image, face_box):
    """
    Heuristic beard detection based on darker lower face region.
    Returns True if a beard is likely present.
    """
    x, y, w, h = face_box
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Define top and bottom face areas
    face_region = gray[y:y+h, x:x+w]
    top_half = face_region[: h//2, :]
    bottom_half = face_region[h//2:, :]

    # Average brightness
    top_mean = np.mean(top_half)
    bottom_mean = np.mean(bottom_half)

    # If bottom half is significantly darker → likely beard
    return bottom_mean < top_mean * 0.8  # 20% darker threshold

# ---- CROP FUNCTION ----
def crop_face_region(image, face_box, beard=False):
    """
    Crops head + neck or head + beard depending on `beard` flag.
    Leaves 2 cm space above head, 2 cm below beard if applicable.
    """
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # cm → pixels
    dpi = 300
    cm2px = dpi / 2.54
    top_margin = int(2 * cm2px)
    bottom_margin = int(2 * cm2px) if beard else int(h * 0.1)

    # Crop coordinates
    x1 = max(x - w // 8, 0)
    x2 = min(x + w + w // 8, img_w)
    y1 = max(y - top_margin, 0)
    y2 = min(y + h + bottom_margin, img_h)

    cropped = image.crop((x1, y1, x2, y2))

    # Maintain 4:5 ratio
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

# ---- MAIN APP LOGIC ----
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)

    with st.spinner("🪄 Processing photo..."):
        face_box = detect_face(image)
        if face_box is None:
            st.error("😕 No face detected. Please upload a clear front-facing portrait photo.")
        else:
            face_box = tuple(map(int, face_box))

            beard = has_beard(image, face_box)
            st.write(f"🧔 Beard detected: **{'Yes' if beard else 'No'}**")

            cropped = crop_face_region(image, face_box, beard=beard)
            final = replace_background_with_white(cropped)
            final = ImageOps.autocontrast(final)

            # Resize to passport size
            final = final.resize((630, 810), Image.LANCZOS)

            st.image(final, caption="✅ Passport-Ready Photo (630×810 px)", use_container_width=True)

            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download Passport Photo",
                data=buf.getvalue(),
                file_name="passport_photo.jpg",
                mime="image/jpeg"
            )
else:
    st.info("👆 Upload a clear, front-facing portrait photo.")
