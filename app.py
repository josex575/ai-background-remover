# app.py
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import cv2
import io
import math

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")
st.title("AI Passport Photo Maker — OpenCV (Cloud-friendly)")
st.markdown("Auto-crop + background removal (OpenCV GrabCut), manual crop sliders, sizes & 4×6 print sheet. No heavy ML libs required.")

# ---------------- Requirements note for README (not used by code) ----------------
# requirements.txt should include:
# streamlit
# opencv-python-headless
# numpy
# Pillow

# ---------------- UI Options ----------------
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

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# ---------------- Face & Eyes Detection ----------------
def detect_face(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    # return largest face
    return max(faces, key=lambda r: r[2] * r[3])

def detect_eyes(pil_img, face_box):
    x, y, w, h = face_box
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    roi = gray[y:y+int(h*0.6), x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 5)
    if len(eyes) == 0:
        return None
    centers = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew//2
        cy = y + ey + eh//2
        centers.append((cx, cy))
    centers.sort(key=lambda p: p[0])
    if len(centers) == 1:
        return (centers[0], centers[0])
    return (centers[0], centers[-1])

# ---------------- Smart AI Crop (adaptive) ----------------
def ai_crop(pil_img, face_box, subject_type, beard=False):
    x, y, w, h = face_box
    img_w, img_h = pil_img.size

    # default margins (fractions)
    top = int(h * 0.9)
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

    cropped = pil_img.crop((x1, y1, x2, y2))

    # pad to 4:5 ratio by adding white margins
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

# ---------------- Background removal using GrabCut ----------------
def grabcut_remove_background(pil_img, face_box):
    """
    Uses OpenCV GrabCut initialized from rectangle around subject (face_box expanded).
    Returns PIL image composited on pure white background.
    """
    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]

    # Expand rectangle around face_box to cover subject body area
    fx, fy, fw, fh = face_box
    # Expand by multipliers
    pad_x = int(fw * 0.9)
    pad_y_top = int(fh * 1.0)
    pad_y_bottom = int(fh * 1.2)
    rx1 = max(0, fx - pad_x)
    ry1 = max(0, fy - pad_y_top)
    rx2 = min(w - 1, fx + fw + pad_x)
    ry2 = min(h - 1, fy + fh + pad_y_bottom)

    rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)

    mask = np.zeros(arr.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # fallback: return original composited on white
        white = np.ones_like(arr) * 255
        return Image.fromarray(white)

    # mask: probable/definite foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    fg = arr * mask2[:, :, np.newaxis]

    # apply small morphological operations to clean edges
    kernel = np.ones((3,3), np.uint8)
    mask_morph = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_DILATE, kernel, iterations=1)

    # feather the mask a bit for hair softness
    mask_f = cv2.GaussianBlur(mask_morph.astype(np.float32), (7,7), 0)
    mask_f = np.clip(mask_f, 0, 1)[..., np.newaxis]

    white_bg = np.ones_like(arr, dtype=np.uint8) * 255
    composite = (arr * mask_f + white_bg * (1 - mask_f)).astype(np.uint8)

    return Image.fromarray(composite)

# ---------------- Auto-contrast and mild sharpening ----------------
def enhance_image(pil_img):
    # autocontrast
    img = ImageOps.autocontrast(pil_img, cutoff=1)
    # mild unsharp mask to reduce perceived blur
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

# ---------------- Eye-based centering ----------------
def center_on_eyes(pil_img, eyes):
    if eyes is None:
        return pil_img
    (lx, ly), (rx, ry) = eyes
    mid_x = int(round((lx + rx) / 2))
    mid_y = int(round((ly + ry) / 2))
    w, h = pil_img.size
    target_y = int(round(h * 0.40))
    dy = mid_y - target_y
    canvas = Image.new("RGB", (w, h), (255,255,255))
    paste_y = -dy
    canvas.paste(pil_img, (0, paste_y))
    return canvas

# ---------------- thin cutline border ----------------
def add_cutline(pil_img, color=(120,120,120), width=1):
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
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

# ---------------- Manual crop UI using sliders ----------------
def manual_crop_with_sliders(pil_img):
    st.subheader("Manual crop (sliders)")
    w, h = pil_img.size
    st.write("Use sliders to position and size the crop rectangle (preview updates).")
    max_w = w
    max_h = h
    left = st.slider("Left (px)", 0, w-1, 0)
    top = st.slider("Top (px)", 0, h-1, 0)
    crop_w = st.slider("Crop width (px)", 50, w-left, min(300, w-left))
    crop_h = st.slider("Crop height (px)", 50, h-top, min(300, h-top))
    crop = pil_img.crop((left, top, left+crop_w, top+crop_h))
    st.image(crop, caption="Manual crop preview", use_column_width=True)
    return crop

# ---------------- Main app flow ----------------
if uploaded is None:
    st.info("Upload a photo to start.")
else:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Original")
    st.image(original, use_column_width=True)

    # ---------- AI passport photo ----------
    st.header("1) AI Passport Photo (Auto) — 630×810 px")
    face = detect_face(original)
    if face is None:
        st.error("Face not detected — AI auto crop cannot proceed. Use manual crop instead.")
    else:
        beard_flag = (photo_type == "With Beard")
        ai_cropped = ai_crop(original, face, subject_type, beard=beard_flag)

        # detect eyes inside cropped image
        inner_face = detect_face(ai_cropped)
        eyes = None
        if inner_face is not None:
            eyes = detect_eyes(ai_cropped, inner_face)

        # background removal via GrabCut (works decently for hair; we apply feathering)
        bg_removed = grabcut_remove_background(ai_cropped, inner_face if inner_face is not None else face)

        # enhancement
        enhanced = enhance_image(bg_removed)

        # center on eyes if available
        centered = center_on_eyes(enhanced, eyes)

        # final resize
        final_ai = centered.resize((630, 810), Image.LANCZOS)

        # add thin cutline
        final_ai = add_cutline(final_ai, color=(120,120,120), width=1)

        st.image(final_ai, caption="AI Passport Photo (630×810 px)", use_column_width=True)

        buf = io.BytesIO(); final_ai.save(buf, "JPEG", quality=95)
        st.download_button("💾 Download AI Passport Photo (630×810)", buf.getvalue(), file_name="passport_ai_630x810.jpg", mime="image/jpeg")

    # ---------- Manual crop ----------
    st.header("2) Manual Crop (Original — no editing applied)")

    manual_crop = manual_crop_with_sliders(original)

    if manual_crop:
        # downloads
        m630 = manual_crop.resize((630, 810), Image.LANCZOS)
        buf1 = io.BytesIO(); m630.save(buf1, "JPEG", quality=95)
        st.download_button("💾 Download Manual Crop (630×810)", buf1.getvalue(), file_name="manual_630x810.jpg", mime="image/jpeg")

        # target chosen size for manual crop
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

        # 2×2 inch @300 dpi option
        buf3 = io.BytesIO()
        manual_crop.resize((600,600), Image.LANCZOS).save(buf3, "JPEG", dpi=(300,300), quality=95)
        st.download_button("💾 Download Manual Crop (2×2 inch @300 DPI)", buf3.getvalue(), file_name="manual_2x2_300dpi.jpg", mime="image/jpeg")

    # ---------- 4x6 print sheet ----------
    if make_sheet:
        st.header("3) 4×6 Print Sheet Generator")
        st.write("Choose source image for tiling:")
        sheet_source = st.radio("Source:", ("AI passport photo (630×810)", "Manual crop (un-edited)"))
        unit_img = None
        unit_w = unit_h = None
        if sheet_source.startswith("AI"):
            if 'final_ai' in locals():
                unit_img = final_ai
                unit_w, unit_h = final_ai.size
            else:
                st.error("AI photo not available.")
        else:
            if manual_crop is None:
                st.error("Make a manual crop first.")
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
            st.image(sheet, caption=f"4×6 sheet ({unit_w}×{unit_h}px tiles) at {dpi} DPI", use_column_width=True)
            buf_sheet = io.BytesIO(); sheet.save(buf_sheet, "JPEG", dpi=(dpi,dpi), quality=95)
            st.download_button("💾 Download 4×6 Print Sheet (JPEG)", buf_sheet.getvalue(), file_name=f"print_sheet_4x6_{dpi}dpi.jpg", mime="image/jpeg")
