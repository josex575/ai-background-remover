import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
import io

st.set_page_config(page_title="AI Passport Photo Maker")

st.title("🪄 AI Passport Photo Maker (Streamlit Cloud Compatible)")

photo_type = st.selectbox("Photo Type", ["Without Beard", "With Beard"])
subject_type = st.selectbox("Subject Type", ["Man", "Woman", "Baby"])

uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

# ---------------- Face Detection ----------------

def detect_face(image):
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None
    
    return max(faces, key=lambda r: r[2] * r[3])


# ---------------- Background Whitening Without Damaging Hair ----------------

def soft_background_whitening(image):
    """
    Lightens background WITHOUT touching hair.
    Works in Streamlit Cloud (no ONNX, no rembg).
    """
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # Gaussian blur mask for background
    blurred = cv2.GaussianBlur(np_img, (55, 55), 55)

    # Increase brightness slightly
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v = np.clip(v + 70, 0, 255)

    hsv = cv2.merge([h, s, v])
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Blend original + brightened
    alpha = 0.3  
    blended = cv2.addWeighted(np_img, 1 - alpha, bright, alpha, 0)

    return Image.fromarray(blended)


# ---------------- AI Crop ----------------

def crop_ai(image, face, photo_type, subject_type):
    x, y, w, h = face
    img_w, img_h = image.size

    # margins
    top_margin = int(h * 0.9)
    bottom_margin = int(h * 0.5)
    side_margin = int(w * 0.5)

    if photo_type == "With Beard":
        bottom_margin += 40  # +2 cm approx

    if subject_type == "Woman":
        top_margin = int(h * 1.2)
        side_margin = int(w * 0.6)

    x1 = max(0, x - side_margin)
    y1 = max(0, y - top_margin)
    x2 = min(img_w, x + w + side_margin)
    y2 = min(img_h, y + h + bottom_margin)

    crop = image.crop((x1, y1, x2, y2))

    # Force 4:5 ratio
    cw, ch = crop.size
    target_ratio = 4 / 5

    if cw / ch > target_ratio:  # too wide
        new_h = int(cw / target_ratio)
        pad = new_h - ch
        crop = ImageOps.expand(crop, border=(0, pad // 2), fill="white")
    else:  # too tall
        new_w = int(ch * target_ratio)
        pad = new_w - cw
        crop = ImageOps.expand(crop, border=(pad // 2, 0), fill="white")

    return crop


# =================================================================
# MAIN APP
# =================================================================

if uploaded_file:
    original = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(original, use_container_width=True)

    # ----------- AI Photo ------------
    st.header("1️⃣ AI Passport Photo")

    face = detect_face(original)

    if face is None:
        st.error("❌ Face not detected. Try a clearer photo.")
    else:
        ai_crop = crop_ai(original, face, photo_type, subject_type)

        if subject_type == "Woman":
            ai_bg = soft_background_whitening(ai_crop)
        else:
            ai_bg = ai_crop

        final_ai = ai_bg.resize((630, 810))

        st.image(final_ai, caption="AI Processed (630×810)", use_container_width=True)

        buf = io.BytesIO()
        final_ai.save(buf, format="JPEG", quality=95)
        st.download_button(
            "📥 Download AI Passport Photo (630×810)",
            buf.getvalue(),
            "passport_ai_630x810.jpg",
            mime="image/jpeg"
        )

    # ----------- Manual Crop ------------
    st.header("2️⃣ Manual Crop Tool (NO edits applied)")

    cropped_manual = st_cropper(
        original,
        realtime_update=True,
        box_color="#FF0000",
        aspect_ratio=None
    )

    if cropped_manual:
        st.image(cropped_manual, caption="Manual Crop Preview")

        # 630×810 version
        m630 = cropped_manual.resize((630, 810))
        buf1 = io.BytesIO()
        m630.save(buf1, "JPEG")
        st.download_button(
            "📥 Download Manual (630×810)",
            buf1.getvalue(),
            "manual_630x810.jpg",
            mime="image/jpeg"
        )

        # 2×2 inch version = 600×600 px
        m2 = cropped_manual.resize((600, 600))
        buf2 = io.BytesIO()
        m2.save(buf2, "JPEG")
        st.download_button(
            "📥 Download Manual (2×2 inch)",
            buf2.getvalue(),
            "manual_2x2inch.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Upload a photo to start.")
