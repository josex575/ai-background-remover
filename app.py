import streamlit as st
from rembg import remove
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import io
import cv2

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker")
st.markdown("""
Upload a portrait photo and choose the correct options below.  
The AI will crop, clean, and prepare passport-style photos (630×810 px and 2×2 inch @ 300 DPI).
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

# ---- SMART CROPPING ----
def crop_based_on_type(image, face_box, photo_type, subject_type):
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    dpi = 300
    cm2px = dpi / 2.54
    top_margin = int(2 * cm2px)

    if subject_type == "Man":
        bottom_margin = int(h * 0.12)
        side_margin = w // 8
    elif subject_type == "Woman":
        top_margin = int(h * 0.6)
        bottom_margin = int(h * 0.25)
        side_margin = int(w * 0.25)
    elif subject_type == "Baby":
        top_margin = int(h * 0.3)
        bottom_margin = int(h * 0.25)
        side_margin = int(w * 0.2)
    else:
        bottom_margin = int(h * 0.1)
        side_margin = w // 8

    if photo_type == "With Beard":
        bottom_margin += int(2 * cm2px)

    x1 = max(x - side_margin, 0)
    x2 = min(x + w + side_margin, img_w)
    y1 = max(y - top_margin, 0)
    y2 = min(y + h + bottom_margin, img_h)

    cropped = image.crop((x1, y1, x2, y2))

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

# ---- BACKGROUND CLEANING ----
def clean_background(image, subject_type):
    """For women → whiten softly without cutting hair; for others → full background removal."""
    np_img = np.array(image.convert("RGB"))

    if subject_type == "Woman":
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.add(L, 25)
        L = np.clip(L, 0, 255)
        lab = cv2.merge((L, A, B))
        brightened = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        mask = cv2.inRange(brightened, (180, 180, 180), (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)
        white_bg = np.full_like(brightened, 255)
        cleaned = cv2.addWeighted(brightened, 0.9, white_bg, 0.1, 0)
        result = np.where(mask_inv[..., None] > 0, cleaned, brightened)
        return Image.fromarray(result.astype(np.uint8))
    else:
        removed = remove(image)
        np_removed = np.array(removed)

        if np_removed.shape[2] == 4:
            alpha = np_removed[:, :, 3]
            white_bg = np.ones_like(np_removed[:, :, :3]) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            composite = white_bg * (1 - alpha_factor) + np_removed[:, :, :3] * alpha_factor
            return Image.fromarray(composite.astype(np.uint8))
        else:
            return image

# ---- ADD THIN CUT-LINE BORDER ----
def add_thin_border(image, line_color=(100, 100, 100), line_width=1):
    bordered = image.copy()
    draw = ImageDraw.Draw(bordered)
    w, h = bordered.size
    draw.rectangle(
        [(line_width // 2, line_width // 2), (w - line_width // 2, h - line_width // 2)],
        outline=line_color,
        width=line_width
    )
    return bordered

# ---- MAIN APP ----
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Original Photo", use_container_width=True)

    with st.spinner("🪄 Processing photo..."):
        face_box = detect_face(image)
        if face_box is None:
            st.error("😕 No face detected. Please upload a clear front-facing portrait photo.")
        else:
            face_box = tuple(map(int, face_box))

            cropped = crop_based_on_type(image, face_box, photo_type, subject_type)
            final = clean_background(cropped, subject_type)
            final = ImageOps.autocontrast(final)

            # --- Photo 1: 630×810 px ---
            final_630x810 = final.resize((630, 810), Image.LANCZOS)
            st.image(final_630x810, caption=f"✅ {subject_type} ({photo_type}) Passport Photo (630×810 px)", use_container_width=True)

            buf = io.BytesIO()
            final_630x810.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download 630×810 px Passport Photo",
                data=buf.getvalue(),
                file_name=f"{subject_type.lower()}_{photo_type.lower().replace(' ', '_')}_passport_photo_630x810.jpg",
                mime="image/jpeg"
            )

            # --- Photo 2: 2×2 inch (600×600 px @ 300 DPI) with thin cut-line border ---
            final_2x2 = final.resize((600, 600), Image.LANCZOS)
            final_with_border = add_thin_border(final_2x2, line_color=(100, 100, 100), line_width=1)

            st.image(final_with_border, caption="✂️ 2×2 inch Photo (600×600 px @ 300 DPI) with Thin Border", use_container_width=True)

            buf_border = io.BytesIO()
            final_with_border.save(buf_border, format="JPEG", quality=95, dpi=(300, 300))
            st.download_button(
                label="💾 Download 2×2 inch Photo (With Thin Border)",
                data=buf_border.getvalue(),
                file_name=f"{subject_type.lower()}_{photo_type.lower().replace(' ', '_')}_2x2_border_photo.jpg",
                mime="image/jpeg"
            )
else:
    st.info("👆 Upload a clear, front-facing portrait photo.")
