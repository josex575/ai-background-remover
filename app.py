import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from streamlit_cropper import st_cropper
import io
import mediapipe as mp
import math

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")
st.title("🪄 AI Passport Photo Maker — Clean White Background (MediaPipe)")
st.markdown("Auto & manual crop. Features: MediaPipe background removal (pure white), auto-centering by eyes, country size selector, 4×6 print sheet, thin cut lines.")

# ---------------- UI ----------------
photo_type = st.selectbox("Photo Type", ["Without Beard", "With Beard"])
subject_type = st.selectbox("Subject Type", ["Man", "Woman", "Baby"])

size_options = {
    "US – 2×2 in (50.8×50.8 mm)": (50.8, 50.8),
    "UK / EU / IN – 35×45 mm": (35.0, 45.0),
    "Canada – 50×70 mm": (50.0, 70.0),
    "Custom (enter mm)": None,
}
size_choice = st.selectbox("Choose target photo size", list(size_options.keys()))
custom_w_mm = custom_h_mm = None
if size_options[size_choice] is None:
    custom_w_mm = st.number_input("Width (mm)", min_value=10.0, max_value=200.0, value=35.0)
    custom_h_mm = st.number_input("Height (mm)", min_value=10.0, max_value=300.0, value=45.0)

dpi = st.selectbox("DPI for print (used for pixel conversions)", [300, 350, 400], index=0)

make_sheet = st.checkbox("Generate 4×6 inch print sheet (selected DPI)", value=True)

uploaded = st.file_uploader("Upload a clear front-facing portrait photo (jpg/png)", type=["jpg", "jpeg", "png"])

# ---------------- Helpers ----------------
def mm_to_px(mm, dpi):
    inches = mm / 25.4
    return int(round(inches * dpi))

def detect_face(image_pil):
    np_img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

def detect_eyes(image_pil, face_box):
    x, y, w, h = face_box
    np_img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    roi_gray = gray[y:y+int(h*0.6), x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    if len(eyes) == 0:
        return None
    abs_eyes = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew//2
        cy = y + ey + eh//2
        abs_eyes.append((cx, cy))
    abs_eyes.sort(key=lambda p: p[0])
    if len(abs_eyes) == 1:
        return (abs_eyes[0], abs_eyes[0])
    return (abs_eyes[0], abs_eyes[-1])

# ---------------- Adaptive AI Crop ----------------
def ai_crop(image_pil, face_box, subject_type, beard=False):
    x, y, w, h = face_box
    img_w, img_h = image_pil.size

    top = int(h * 0.90)
    bottom = int(h * 0.45)
    sides = int(w * 0.55)

    if beard:
        bottom += int(h * 0.25)

    if subject_type == "Woman":
        top = int(h * 1.2)
        sides = int(w * 0.7)
        bottom = int(h * 0.45)
    if subject_type == "Baby":
        top = int(h * 0.7)
        sides = int(w * 0.45)
        bottom = int(h * 0.5)

    x1 = max(0, x - sides)
    y1 = max(0, y - top)
    x2 = min(img_w, x + w + sides)
    y2 = min(img_h, y + h + bottom)

    cropped = image_pil.crop((x1, y1, x2, y2))

    # ensure 4:5 ratio by padding with white
    cw, ch = cropped.size
    target_ratio = 4 / 5
    if cw / ch > target_ratio:
        new_h = int(round(cw / target_ratio))
        pad = new_h - ch
        cropped = ImageOps.expand(cropped, border=(0, pad//2), fill="white")
    else:
        new_w = int(round(ch * target_ratio))
        pad = new_w - cw
        cropped = ImageOps.expand(cropped, border=(pad//2, 0), fill="white")
    return cropped

# ---------------- MediaPipe segmentation-based background removal ----------------
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background_mediapipe(pil_img, threshold=0.5, pure_white=True):
    """
    Uses Mediapipe SelfieSegmentation to produce a clean mask and composite a pure white background.
    threshold: mask threshold [0..1]
    pure_white: if True, set background to pure white
    """
    rgb = np.array(pil_img.convert("RGB"))
    # Mediapipe expects BGR input
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = mp_selfie.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if results.segmentation_mask is None:
        # fallback: return original
        return pil_img
    mask = results.segmentation_mask  # float mask [0..1]
    # binary-ish mask with soft edges:
    mask3 = np.dstack([mask]*3)
    if pure_white:
        white = np.ones_like(rgb, dtype=np.uint8) * 255
        composite = (rgb * mask3 + white * (1 - mask3)).astype(np.uint8)
    else:
        # slight feathering toward white
        white = np.ones_like(rgb, dtype=np.uint8) * 255
        composite = (rgb * mask3 + white * (1 - mask3*0.8)).astype(np.uint8)
    return Image.fromarray(composite)

# ---------------- Auto-contrast ----------------
def auto_contrast(pil_img):
    return ImageOps.autocontrast(pil_img, cutoff=1)

# ---------------- Eye-based centering ----------------
def center_on_eyes(pil_img, eyes):
    """Shift canvas vertically so eyes midpoint sits near 40% from top."""
    if eyes is None:
        return pil_img
    (lx, ly), (rx, ry) = eyes
    mid_x = int(round((lx + rx) / 2))
    mid_y = int(round((ly + ry) / 2))
    w, h = pil_img.size
    # target y position (40% down)
    target_y = int(round(h * 0.40))
    dy = mid_y - target_y
    canvas = Image.new("RGB", (w, h), (255,255,255))
    paste_y = -dy
    canvas.paste(pil_img, (0, paste_y))
    return canvas

# ---------------- thin trim border ----------------
def add_trim_border(pil_img, color=(120,120,120), width=1):
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    # draw rectangle inset by half a pixel to keep 1px visible
    draw.rectangle([0.5, 0.5, w-0.5, h-0.5], outline=color, width=width)
    return out

# ---------------- print sheet generator ----------------
def generate_print_sheet(single_img_pil, unit_w, unit_h, sheet_inches=(4,6), dpi=300):
    sheet_w = int(round(sheet_inches[0]*dpi))
    sheet_h = int(round(sheet_inches[1]*dpi))
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255,255,255))
    cols = sheet_w // unit_w
    rows = sheet_h // unit_h
    draw = ImageDraw.Draw(sheet)
    if cols == 0 or rows == 0:
        x = (sheet_w - unit_w)//2
        y = (sheet_h - unit_h)//2
        sheet.paste(single_img_pil, (x,y))
        draw.rectangle([x+0.5, y+0.5, x+unit_w-0.5, y+unit_h-0.5], outline=(120,120,120), width=1)
        return sheet
    for r in range(rows):
        for c in range(cols):
            x = c*unit_w
            y = r*unit_h
            sheet.paste(single_img_pil, (x,y))
            draw.rectangle([x+0.5, y+0.5, x+unit_w-0.5, y+unit_h-0.5], outline=(120,120,120), width=1)
    return sheet

# ---------------- Manual cropping helper (streamlit-cropper will be used later) ----------------

# ---------------- Main ----------------
if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Original Photo")
    st.image(original, use_column_width=True)

    # AI Passport Photo
    st.header("1️⃣ AI Passport Photo — Auto (630×810 px)")

    face = detect_face(original)
    if face is None:
        st.error("No face detected. Use manual crop or upload a clearer photo.")
    else:
        beard_flag = (photo_type == "With Beard")
        cropped = ai_crop(original, face, subject_type, beard=beard_flag)

        # detect eyes inside the cropped area to center precisely
        inner_face = detect_face(cropped)
        eyes = None
        if inner_face is not None:
            eyes = detect_eyes(cropped, inner_face)

        # MediaPipe background removal -> pure white background (option A)
        processed = remove_background_mediapipe(cropped, threshold=0.5, pure_white=True)

        # auto-contrast
        processed = auto_contrast(processed)

        # center on eyes
        processed = center_on_eyes(processed, eyes)

        # resize to passport size
        final_ai = processed.resize((630, 810), Image.LANCZOS)

        # thin trim border
        final_ai = add_trim_border(final_ai, color=(120,120,120), width=1)

        st.image(final_ai, caption="AI Passport Photo (630×810 px)", use_column_width=True)

        buf = io.BytesIO()
        final_ai.save(buf, format="JPEG", quality=95)
        st.download_button("💾 Download AI Passport Photo (630×810)", buf.getvalue(), file_name="passport_ai_630x810.jpg", mime="image/jpeg")

    # Manual Crop Tool (unchanged)
    st.header("2️⃣ Manual Crop Tool (Original image, no edits)")
    st.write("Drag the box to crop the original. Downloads are unedited except for resizing.")
    manual_crop = st_cropper(original, realtime_update=True, box_color="#FF0000", aspect_ratio=None)

    if manual_crop:
        st.image(manual_crop, caption="Manual Crop Preview", use_column_width=True)

        m630 = manual_crop.resize((630, 810), Image.LANCZOS)
        buf1 = io.BytesIO(); m630.save(buf1, "JPEG", quality=95)
        st.download_button("💾 Download Manual Crop (630×810)", buf1.getvalue(), file_name="manual_630x810.jpg", mime="image/jpeg")

        # target chosen size
        if size_options[size_choice] is not None:
            w_mm, h_mm = size_options[size_choice]
        else:
            w_mm, h_mm = custom_w_mm, custom_h_mm
        target_w_px = mm_to_px(w_mm, dpi)
        target_h_px = mm_to_px(h_mm, dpi)
        manual_target = manual_crop.resize((target_w_px, target_h_px), Image.LANCZOS)
        buf2 = io.BytesIO(); manual_target.save(buf2, "JPEG", dpi=(dpi,dpi), quality=95)
        st.download_button(f"💾 Download Manual Crop ({int(w_mm)}×{int(h_mm)} mm @ {dpi} DPI)", buf2.getvalue(),
                           file_name=f"manual_{int(w_mm)}x{int(h_mm)}mm_{dpi}dpi.jpg", mime="image/jpeg")

        if st.button("Create 2×2 inch (600×600 px @300 DPI) from manual crop"):
            m2 = manual_crop.resize((600, 600), Image.LANCZOS)
            buf3 = io.BytesIO(); m2.save(buf3, "JPEG", dpi=(300,300), quality=95)
            st.download_button("💾 Download Manual 2×2 inch", buf3.getvalue(), file_name="manual_2x2_300dpi.jpg", mime="image/jpeg")

    # 4x6 print sheet
    if make_sheet:
        st.header("3️⃣ 4×6 Print Sheet")
        st.write("Choose which image to tile on the 4×6 sheet:")

        sheet_source = st.radio("Source for sheet:", ("AI passport photo (630×810)", "Manual crop (un-edited)"))
        unit_img = None
        if sheet_source.startswith("AI"):
            if 'final_ai' not in locals():
                st.error("AI photo not available (face detection may have failed).")
            else:
                unit_img = final_ai
                unit_w, unit_h = unit_img.size
        else:
            if manual_crop is None:
                st.error("Please make a manual crop first.")
            else:
                if size_options[size_choice] is not None:
                    w_mm, h_mm = size_options[size_choice]
                else:
                    w_mm, h_mm = custom_w_mm, custom_h_mm
                unit_w = mm_to_px(w_mm, dpi)
                unit_h = mm_to_px(h_mm, dpi)
                unit_img = manual_crop.resize((unit_w, unit_h), Image.LANCZOS)

        if unit_img is not None:
            sheet = generate_print_sheet(unit_img, unit_w, unit_h, sheet_inches=(4,6), dpi=dpi)
            st.image(sheet, caption=f"4×6 sheet with tiles ({unit_w}×{unit_h}px) at {dpi} DPI", use_column_width=True)
            buf_sheet = io.BytesIO(); sheet.save(buf_sheet, "JPEG", dpi=(dpi,dpi), quality=95)
            st.download_button("💾 Download 4×6 Print Sheet (JPEG)", buf_sheet.getvalue(), file_name=f"print_sheet_4x6_{dpi}dpi.jpg", mime="image/jpeg")

else:
    st.info("Upload a photo to begin.")
