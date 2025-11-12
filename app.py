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
The AI will crop, clean, and prepare a passport-style photo (630×810 px).
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
    """
    Crops image based on subject type:
    - Man: normal head + neck
    - With Beard: adds 2 cm below beard
    - Woman: looser top/sides to include hair
    - Baby: extra space around head
    """
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

# ---- BACKGROUND REMOVAL ----
def replace_background_with_white(image, subject_type):
    """Removes background. For women: soften the edges around hair."""
    removed = remove(image)
    np_img = np.array(removed)

    if np_img.shape[2] == 4:
        alpha = np_img[:, :, 3]
        if subject_type == "Woman":
            alpha = cv2.GaussianBlur(alpha, (7, 7), 3)

        white_bg = np.ones_like(np_img[:, :, :3]) * 255
        alpha_factor = alpha[:, :, np.newaxis] / 255.0
        composite = white_bg * (1 - alpha_factor) + np_img[:, :, :3] * alpha_factor
        return Image.fromarray(composite.astype(np.uint8))
    else:
        return image

# ---- ADD THIN CUT LINE BORDER ----
def add_thin_border(image, line_color=(150, 150, 150), line_width=2):
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
            final = replace_background_with_white(cropped, subject_type)
            final = ImageOps.autocontrast(final)

            # Resize to passport dimensions
            final = final.resize((630, 810), Image.LANCZOS)

            st.image(final, caption=f"✅ {subject_type} ({photo_type}) Passport Photo (630×810 px)", use_container_width=True)

            # ---- Download original photo ----
            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download Passport Photo",
                data=buf.getvalue(),
                file_name=f"{subject_type.lower()}_{photo_type.lower().replace(' ', '_')}_passport_photo.jpg",
                mime="image/jpeg"
            )

            # ---- Add thin cut line border ----
            final_with_border = add_thin_border(final, line_color=(150, 150, 150), line_width=2)

            st.image(final_with_border, caption="✂️ Passport Photo with Thin Print Border", use_container_width=True)

            # Download button for bordered image
            buf_border = io.BytesIO()
            final_with_border.save(buf_border, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download Photo with Thin Border (Cut Line)",
                data=buf_border.getvalue(),
                file_name=f"{subject_type.lower()}_{photo_type.lower().replace(' ', '_')}_cutline_photo.jpg",
                mime="image/jpeg"
            )
else:
    st.info("👆 Upload a clear, front-facing portrait photo.")
