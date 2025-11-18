import streamlit as st
from rembg import remove
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import io
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")

st.title("🪄 AI Passport Photo Maker (AI + Manual Crop Tool)")
st.markdown("""
Upload a portrait photo and choose the options below.  
You will get:

1️⃣ **AI-processed passport photo (630×810 px)**  
2️⃣ **Manual crop tool** → crop original photo yourself, without any edits  
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

uploaded = st.file_uploader("📸 Upload Photo", type=["jpg", "jpeg", "png"])


# ----------------------------------------------------------------------------------------------------------------------
# FACE DETECTION
# ----------------------------------------------------------------------------------------------------------------------
def detect_face(image):
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


# ----------------------------------------------------------------------------------------------------------------------
# SMART AUTO CROP FOR PASSPORT (PHOTO 1)
# ----------------------------------------------------------------------------------------------------------------------
def crop_ai(image, face_box, photo_type, subject_type):
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

    # Keep passport ratio 4:5
    target_ratio = 4 / 5
    cw, ch = cropped.size

    if cw / ch > target_ratio:
        new_h = int(cw / target_ratio)
        pad = new_h - ch
        cropped = ImageOps.expand(cropped, border=(0, pad // 2), fill="white")
    else:
        new_w = int(ch * target_ratio)
        pad = new_w - cw
        cropped = ImageOps.expand(cropped, border=(pad // 2, 0), fill="white")

    return cropped


# ----------------------------------------------------------------------------------------------------------------------
# CLEAN BACKGROUND
# ----------------------------------------------------------------------------------------------------------------------
def clean_background(image, subject_type):
    # For women → preserve hair
    if subject_type == "Woman":
        np_img = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.add(L, 25)
        L = np.clip(L, 0, 255)
        lab = cv2.merge((L, A, B))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    # For others → full background removal
    removed = remove(image)
    np_removed = np.array(removed)

    if np_removed.shape[2] == 4:
        alpha = np_removed[:, :, 3]
        white_bg = np.ones_like(np_removed[:, :, :3]) * 255
        a = alpha[:, :, None] / 255.0
        composite = white_bg * (1 - a) + np_removed[:, :, :3] * a
        return Image.fromarray(composite.astype(np.uint8))

    return image


# ----------------------------------------------------------------------------------------------------------------------
# MANUAL CROP TOOL
# ----------------------------------------------------------------------------------------------------------------------
def interactive_crop(image):
    st.subheader("✂️ Manual Crop Tool (No Editing Applied)")

    w, h = image.size

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_color="red",
        background_image=image,
        update_streamlit=True,
        height=max(400, h),
        width=min(800, w),
        drawing_mode="rect",
        key="canvas_crop"
    )

    if canvas.json_data is None:
        return None

    if "objects" not in canvas.json_data:
        return None

    if len(canvas.json_data["objects"]) == 0:
        return None

    rect = canvas.json_data["objects"][-1]

    left = int(rect["left"])
    top = int(rect["top"])
    width = int(rect["width"])
    height = int(rect["height"])

    crop = image.crop((left, top, left + width, top + height))
    return crop


# ======================================================================================================================
# MAIN UI LOGIC
# ======================================================================================================================
if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.image(original, caption="Original Uploaded Photo", use_container_width=True)

    # ------------------------------------------------------------
    # PHOTO 1: AI PROCESSED PASSPORT PHOTO (630×810)
    # ------------------------------------------------------------
    st.header("📌 AI Passport Photo (Auto-Crop + Background Clean)")

    with st.spinner("Processing AI passport photo…"):
        face_box = detect_face(original)
        if face_box is None:
            st.error("No face detected in photo.")
        else:
            cropped = crop_ai(original, face_box, photo_type, subject_type)
            cleaned = clean_background(cropped, subject_type)
            final_630 = cleaned.resize((630, 810), Image.LANCZOS)

            st.image(final_630, caption="AI Passport Photo (630×810)", use_container_width=True)

            buf = io.BytesIO()
            final_630.save(buf, format="JPEG", quality=95)
            st.download_button(
                "💾 Download AI Passport Photo (630×810)",
                buf.getvalue(),
                file_name="passport_ai_630x810.jpg",
                mime="image/jpeg"
            )

    # ------------------------------------------------------------
    # PHOTO 2: MANUAL CROP TOOL
    # ------------------------------------------------------------
    st.header("✂️ Manual Crop Tool (Based on Original Photo)")

    cropped_manual = interactive_crop(original)

    if cropped_manual:
        st.subheader("📸 Your Manual Crop Result (Unedited)")
        st.image(cropped_manual, use_container_width=True)

        # 630×810 version
        crop_630 = cropped_manual.resize((630, 810), Image.LANCZOS)
        buf1 = io.BytesIO()
        crop_630.save(buf1, "JPEG", quality=95)
        st.download_button(
            "💾 Download Manual Crop (630×810)",
            buf1.getvalue(),
            file_name="manual_crop_630x810.jpg",
            mime="image/jpeg"
        )

        # 2×2 inch (600×600 px @ 300 DPI)
        crop_2x2 = cropped_manual.resize((600, 600), Image.LANCZOS)
        buf2 = io.BytesIO()
        crop_2x2.save(buf2, "JPEG", dpi=(300, 300), quality=95)
        st.download_button(
            "💾 Download Manual Crop (2×2 inch @ 300 DPI)",
            buf2.getvalue(),
            file_name="manual_crop_2x2inch.jpg",
            mime="image/jpeg"
        )

else:
    st.info("Upload an image to begin.")


