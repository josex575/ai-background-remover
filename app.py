import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from streamlit_cropper import st_cropper
import io
import math

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")
st.title("🪄 AI Passport Photo Maker — Auto-center, Sizes, Print Sheet, Cut Lines")
st.markdown("AI passport photo + manual crop. Features: auto-centering by eyes, country size selector, 4×6 print sheet generator and thin cut lines.")

# -------------------- UI Options --------------------
photo_type = st.selectbox("Photo Type", ["Without Beard", "With Beard"])
subject_type = st.selectbox("Subject Type", ["Man", "Woman", "Baby"])

# Country / size selector: maps to (width_mm, height_mm)
size_options = {
    "US – 2×2 in": (50.8, 50.8),        # mm (2 in = 50.8 mm)
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

# 4x6 sheet toggle
make_sheet = st.checkbox("Generate 4×6 inch print sheet (300 DPI) with trim lines", value=True)

# Upload
uploaded = st.file_uploader("Upload a clear front-facing portrait photo (jpg/png)", type=["jpg", "jpeg", "png"])

# -------------------- Helpers --------------------
def mm_to_px(mm, dpi):
    inches = mm / 25.4
    return int(round(inches * dpi))

def detect_face(image_pil):
    np_img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    # return largest
    return max(faces, key=lambda r: r[2] * r[3])

def detect_eyes(image_pil, face_box):
    """
    Returns (left_eye_center, right_eye_center) as (x,y) in image coords if found, else None.
    """
    x, y, w, h = face_box
    np_img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    roi_gray = gray[y:y+int(h*0.7), x:x+w]  # search upper part of face
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    if len(eyes) == 0:
        return None
    # convert to absolute coords and pick two most likely eyes by x position
    abs_eyes = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew//2
        cy = y + ey + eh//2
        abs_eyes.append((cx, cy))
    # sort by x and take leftmost and rightmost
    abs_eyes.sort(key=lambda p: p[0])
    if len(abs_eyes) == 1:
        return (abs_eyes[0], abs_eyes[0])
    else:
        return (abs_eyes[0], abs_eyes[-1])

def ai_crop(image_pil, face_box, subject_type, beard=False):
    """Adaptive crop around face; returns PIL image (cropped, padded to 4:5)."""
    x, y, w, h = face_box
    img_w, img_h = image_pil.size

    # base margins (fractions of face h/w)
    top = int(h * 0.90)
    bottom = int(h * 0.45)
    sides = int(w * 0.55)

    if beard:
        bottom += int(h * 0.25)  # extra space for beard

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

def white_background_soft(image_pil):
    """Lighten background while keeping hair edges natural. Uses blur-based blending."""
    np_img = np.array(image_pil.convert("RGB"))
    blur = cv2.GaussianBlur(np_img, (55, 55), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + 60, 0, 255)
    hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # blend: keep original (foreground) stronger, brightened background slightly
    alpha = 0.25
    blended = cv2.addWeighted(np_img, 1.0, bright, alpha, 0)
    # convert darker background pixels toward white a bit
    lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    # increase low-L areas more
    L2 = np.where(L < 200, np.clip(L + 30, 0, 255), L)
    lab2 = cv2.merge((L2.astype(np.uint8), A, B))
    res = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(res.astype(np.uint8))

def auto_contrast(pil_img):
    return ImageOps.autocontrast(pil_img, cutoff=1)

def center_on_eyes(image_pil, eyes, target_height_px):
    """
    Recenters the image so that eyes midpoint sits at target_height_px (pixel coordinate)
    within the current image. If eyes None, returns original.
    """
    if eyes is None:
        return image_pil
    (lx, ly), (rx, ry) = eyes
    mid_x = int(round((lx + rx) / 2))
    mid_y = int(round((ly + ry) / 2))

    w, h = image_pil.size
    # target vertical position (we choose ratio of frame height)
    # use eyes_target_ratio so eyes sit at ~40% from top
    eyes_target_ratio = 0.40
    target_y = int(round(h * eyes_target_ratio))

    dy = mid_y - target_y  # positive if eyes below target -> shift up
    # create new canvas and paste so that we shift content up/down by dy
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    paste_y = -dy
    # ensure paste_y such that image overlaps canvas correctly
    canvas.paste(image_pil, (0, paste_y))
    return canvas

# -------------------- Print sheet generator --------------------
def generate_print_sheet(single_img_pil, unit_px_width, unit_px_height, sheet_inches=(4,6), dpi=300, draw_cutlines=True):
    """
    single_img_pil: PIL image sized exactly unit_px_width x unit_px_height (px)
    sheet_inches: (width_in, height_in)
    dpi: dots per inch
    returns PIL sheet image (sheet_px_w x sheet_px_h)
    """
    sheet_w = int(round(sheet_inches[0] * dpi))
    sheet_h = int(round(sheet_inches[1] * dpi))
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255,255,255))
    cols = sheet_w // unit_px_width
    rows = sheet_h // unit_px_height
    if cols == 0 or rows == 0:
        # fall back: center single image
        x = (sheet_w - unit_px_width)//2
        y = (sheet_h - unit_px_height)//2
        sheet.paste(single_img_pil, (x, y))
        draw = ImageDraw.Draw(sheet)
        if draw_cutlines:
            draw.rectangle([x+0.5, y+0.5, x+unit_px_width-0.5, y+unit_px_height-0.5], outline=(120,120,120), width=1)
        return sheet

    draw = ImageDraw.Draw(sheet)
    for r in range(rows):
        for c in range(cols):
            x = c * unit_px_width
            y = r * unit_px_height
            sheet.paste(single_img_pil, (x, y))
            if draw_cutlines:
                # thin cut line rectangle (0.5 to avoid being clipped)
                draw.rectangle([x+0.5, y+0.5, x+unit_px_width-0.5, y+unit_px_height-0.5], outline=(120,120,120), width=1)
    return sheet

# -------------------- Main --------------------
if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Original Photo")
    st.image(original, use_column_width=True)

    # ---- AI Passport Photo (auto) ----
    st.header("1️⃣ AI Passport Photo — Auto (630×810 px)")

    face = detect_face(original)
    if face is None:
        st.error("No face detected — AI auto-crop cannot proceed. Use manual crop or upload clearer photo.")
    else:
        beard_flag = (photo_type == "With Beard")
        cropped = ai_crop(original, face, subject_type, beard=beard_flag)

        # detect eyes in cropped coordinates: we need eyes in cropped image space
        # map face_box into cropped coords: recompute by re-detecting face inside cropped (safer)
        inner_face = detect_face(cropped)
        eyes = None
        if inner_face is not None:
            eyes = detect_eyes(cropped, inner_face)

        # white background softly (women safe)
        processed = white_background_soft(cropped)

        # auto-contrast
        processed = auto_contrast(processed)

        # center on eyes if available
        processed = center_on_eyes(processed, eyes, target_height_px=int(processed.size[1]*0.4))

        # final resize to 630x810
        final_ai = processed.resize((630, 810), Image.LANCZOS)

        # add thin cut line around final image (1 px)
        draw = ImageDraw.Draw(final_ai)
        w, h = final_ai.size
        draw.rectangle([0.5, 0.5, w-0.5, h-0.5], outline=(120,120,120), width=1)

        st.image(final_ai, caption="AI Passport Photo (630×810 px) — with thin cut line", use_column_width=True)

        buf = io.BytesIO()
        final_ai.save(buf, format="JPEG", quality=95)
        st.download_button("💾 Download AI Passport Photo (630×810)", buf.getvalue(), file_name="passport_ai_630x810.jpg", mime="image/jpeg")

    # ---- Manual crop tool (unchanged) ----
    st.header("2️⃣ Manual Crop Tool (Original — no edits)")
    st.write("Drag to select area. After cropping you can download unedited versions at chosen sizes.")
    manual_crop = st_cropper(original, realtime_update=True, box_color="#FF0000", aspect_ratio=None)

    if manual_crop:
        st.image(manual_crop, caption="Manual Crop Preview", use_column_width=True)
        # manual downloads: 630x810 and selected country size & 2x2 inch option
        m630 = manual_crop.resize((630, 810), Image.LANCZOS)
        buf1 = io.BytesIO(); m630.save(buf1, "JPEG", quality=95)
        st.download_button("💾 Download Manual Crop (630×810)", buf1.getvalue(), file_name="manual_630x810.jpg", mime="image/jpeg")

        # get target size pixels for selected size_choice
        if size_options[size_choice] is not None:
            w_mm, h_mm = size_options[size_choice]
        else:
            w_mm, h_mm = custom_w_mm, custom_h_mm
        target_w_px = mm_to_px(w_mm, dpi)
        target_h_px = mm_to_px(h_mm, dpi)

        manual_target = manual_crop.resize((target_w_px, target_h_px), Image.LANCZOS)
        buf2 = io.BytesIO(); manual_target.save(buf2, "JPEG", dpi=(dpi, dpi), quality=95)
        st.download_button(f"💾 Download Manual Crop ({int(w_mm)}×{int(h_mm)} mm @ {dpi} DPI)", buf2.getvalue(),
                           file_name=f"manual_{int(w_mm)}x{int(h_mm)}mm_{dpi}dpi.jpg", mime="image/jpeg")

        # extra: make 2x2 inch (600x600 @300 DPI) if user wants
        if st.button("Create 2×2 inch (600×600 @300 DPI) from manual crop"):
            m2 = manual_crop.resize((600, 600), Image.LANCZOS)
            buf3 = io.BytesIO(); m2.save(buf3, "JPEG", dpi=(300,300), quality=95)
            st.download_button("💾 Download Manual 2×2 inch", buf3.getvalue(), file_name="manual_2x2_300dpi.jpg", mime="image/jpeg")

    # ---- 4×6 print sheet generation ----
    if make_sheet and uploaded:
        st.header("3️⃣ 4×6 Print Sheet")
        st.write("This will generate a 4×6 inch (W×H) print sheet at selected DPI with thin cut lines; it tiles the selected single image size.")

        # choose which image to tile: AI or manual?
        sheet_source = st.radio("Select source image for sheet:", ("AI passport photo (630×810)", "Manual crop (un-edited)"))
        if sheet_source.startswith("AI"):
            if 'final_ai' not in locals():
                st.error("AI photo not available (face detection failed).")
            else:
                unit_img = final_ai
                # unit size in px is size of final_ai (630x810)
                unit_w, unit_h = unit_img.size
        else:
            if manual_crop is None:
                st.error("Please make a manual crop first.")
                unit_img = None
            else:
                # use target chosen size or manual crop scaled to target
                if size_options[size_choice] is not None:
                    w_mm, h_mm = size_options[size_choice]
                else:
                    w_mm, h_mm = custom_w_mm, custom_h_mm
                unit_w = mm_to_px(w_mm, dpi)
                unit_h = mm_to_px(h_mm, dpi)
                unit_img = manual_crop.resize((unit_w, unit_h), Image.LANCZOS)

        if unit_img is not None:
            sheet = generate_print_sheet(unit_img, unit_w, unit_h, sheet_inches=(4,6), dpi=dpi, draw_cutlines=True)
            st.image(sheet, caption=f"4×6 sheet w/ {unit_w}×{unit_h}px tiles at {dpi} DPI", use_column_width=True)
            buf_sheet = io.BytesIO(); sheet.save(buf_sheet, "JPEG", dpi=(dpi,dpi), quality=95)
            st.download_button("💾 Download 4×6 Print Sheet (JPEG)", buf_sheet.getvalue(), file_name=f"print_sheet_4x6_{dpi}dpi.jpg", mime="image/jpeg")

else:
    st.info("Upload a photo to begin.")
