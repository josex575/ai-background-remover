import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import cv2
from streamlit_cropper import st_cropper

st.set_page_config(page_title="AI Passport Photo Maker")

st.title("🪄 AI Passport Photo Maker (AI + Manual Crop)")
st.write("Upload a photo to get AI-cropped passport image + manual cropping tool.")

# ------------------------------- OPTIONS -------------------------------
photo_type = st.selectbox("Photo Type", ["Without Beard", "With Beard"])
subject_type = st.selectbox("Subject Type", ["Man", "Woman", "Baby"])

uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])


# ------------------------------- FACE DETECTION -------------------------------
def detect_face(image):
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


# ------------------------------- AI CROP -------------------------------
def crop_ai(image, face_box, photo_type, subject_type):
    x, y, w, h = face_box
    img_w, img_h = image.size

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

    if photo_type == "With Beard":
        bottom_margin += int(2 * cm2px)

    x1 = max(0, x - side_margin)
    x2 = min(img_w, x + w + side_margin)
    y1 = max(0, y - top_margin)
    y2 = min(img_h, y + h + bottom_margin)

    cropped = image.crop((x1, y1, x2, y2))

    # Keep 4:5 ratio
    cw, ch = cropped.size
    target_ratio = 4 / 5

    if cw / ch > target_ratio:  # too wide
        new_h = int(cw / target_ratio)
        pad = new_h - ch
        cropped = ImageOps.expand(cropped, border=(0, pad // 2), fill="white")
    else:  # too tall
        new_w = int(ch * target_ratio)
        pad = new_w - cw
        cropped = ImageOps.expand(cropped, border=(pad // 2, 0), fill="white")

    return cropped


# ------------------------------- WOMEN SAFE BACKGROUND -------------------------------
def clean_background(image, subject_type):
    if subject_type == "Woman":
        np_img = np.array(image)
        import cv2
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.add(L, 25)
        L = np.clip(L, 0, 255)
        merged = cv2.merge((L, A, B))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)

    removed = remove(image)
    np_removed = np.array(removed)

    if np_removed.shape[2] == 4:
        alpha = np_removed[:, :, 3] / 255.0
        white = np.ones_like(np_removed[:, :, :3]) * 255
        blended = white * (1 - alpha[:, :, None]) + np_removed[:, :, :3] * alpha[:, :, None]
        return Image.fromarray(blended.astype(np.uint8))

    return image


# ======================================================================
# MAIN LOGIC
# ======================================================================
if uploaded_file:

    original = Image.open(uploaded_file).convert("RGB")
    st.image(original, caption="Original Photo", use_container_width=True)

    # ------------------------------- AI PHOTO -------------------------------
    st.header("1️⃣ AI Passport Photo (630×810)")

    face = detect_face(original)

    if not face:
        st.error("❌ Could not detect a face. Please upload a clearer photo.")
    else:
        cropped_ai = crop_ai(original, face, photo_type, subject_type)
        cleaned_ai = clean_background(cropped_ai, subject_type)
        final_ai = cleaned_ai.resize((630, 810))

        st.image(final_ai, caption="AI Processed Passport Photo", use_container_width=True)

        buf = io.BytesIO()
        final_ai.save(buf, format="JPEG", quality=95)
        st.download_button(
            "📥 Download AI Passport Photo (630×810)",
            buf.getvalue(),
            "passport_ai_630x810.jpg",
            mime="image/jpeg"
        )

    # ------------------------------- MANUAL CROP TOOL -------------------------------
    st.header("2️⃣ Manual Crop Tool (Original Image – No Edits)")

    st.write("Drag the crop box to select your desired area:")

    manual_crop = st_cropper(
        original,
        realtime_update=True,
        box_color="#FF0000",
        aspect_ratio=None
    )

    if manual_crop:
        st.image(manual_crop, caption="Manual Crop Preview", use_container_width=True)

        # 630×810 version
        m630 = manual_crop.resize((630, 810))
        buf1 = io.BytesIO()
        m630.save(buf1, "JPEG")
        st.download_button(
            "📥 Download Manual Crop (630×810)",
            buf1.getvalue(),
            "manual_630x810.jpg",
            mime="image/jpeg"
        )

        # 2×2 inch = 600×600 px
        m2in = manual_crop.resize((600, 600))
        buf2 = io.BytesIO()
        m2in.save(buf2, "JPEG", dpi=(300, 300))
        st.download_button(
            "📥 Download Manual Crop (2×2 inch @300 DPI)",
            buf2.getvalue(),
            "manual_2x2inch.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Please upload a photo to continue.")
