import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import cv2
import io

st.set_page_config(page_title="AI Passport Photo Maker", layout="centered")
st.title("AI Passport Photo Maker — Cloud-friendly")
st.markdown("Auto-crop + background removal using OpenCV + manual crop sliders. No incompatible libraries.")

# ---------------- UI ----------------
photo_type = st.selectbox("Photo Type", ["Without Beard", "With Beard"])
subject_type = st.selectbox("Subject Type", ["Man", "Woman", "Baby"])
uploaded = st.file_uploader("Upload your portrait photo", type=["jpg", "jpeg", "png"])

# ---------------- Helpers ----------------
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def detect_face(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2]*r[3])

def ai_crop(img, face_box, subject_type, beard=False):
    x, y, w, h = face_box
    img_w, img_h = img.size

    # Default margins
    top = int(h*0.9)
    bottom = int(h*0.45)
    sides = int(w*0.55)
    if beard:
        bottom += int(h*0.25)
    if subject_type == "Woman":
        top = int(h*1.2)
        sides = int(w*0.7)
        bottom = int(h*0.45)
    if subject_type == "Baby":
        top = int(h*0.7)
        sides = int(w*0.45)
        bottom = int(h*0.5)

    x1 = max(0, x-sides)
    y1 = max(0, y-top)
    x2 = min(img_w, x+w+sides)
    y2 = min(img_h, y+h+bottom)
    cropped = img.crop((x1, y1, x2, y2))

    # Pad to 4:5 ratio
    cw, ch = cropped.size
    target_ratio = 4/5
    if cw/ch > target_ratio:
        new_h = int(round(cw/target_ratio))
        pad = new_h - ch
        cropped = ImageOps.expand(cropped, border=(0,pad//2), fill="white")
    else:
        new_w = int(round(ch*target_ratio))
        pad = new_w - cw
        cropped = ImageOps.expand(cropped, border=(pad//2,0), fill="white")
    return cropped

def grabcut_bg(img, face_box):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    fx, fy, fw, fh = face_box
    pad_x = int(fw*0.9)
    pad_y_top = int(fh*1.0)
    pad_y_bottom = int(fh*1.2)
    rect = (max(0, fx-pad_x), max(0, fy-pad_y_top), min(w-1, fx+fw+pad_x)-max(0, fx-pad_x), min(h-1, fy+fh+pad_y_bottom)-max(0, fy-pad_y_top))
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except:
        return Image.new("RGB", img.size, "white")
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    fg = arr*mask2[:,:,np.newaxis]
    # Feather edges
    mask_f = cv2.GaussianBlur(mask2.astype(np.float32),(7,7),0)[...,np.newaxis]
    white_bg = np.ones_like(arr)*255
    comp = (arr*mask_f + white_bg*(1-mask_f)).astype(np.uint8)
    return Image.fromarray(comp)

def enhance(img):
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

def add_border(img, color=(120,120,120), width=1):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    draw.rectangle([0.5,0.5,w-0.5,h-0.5], outline=color, width=width)
    return img

def manual_crop(img):
    st.subheader("Manual Crop (Original)")
    w,h = img.size
    left = st.slider("Left",0,w-1,0)
    top = st.slider("Top",0,h-1,0)
    crop_w = st.slider("Width",50,w-left,min(300,w-left))
    crop_h = st.slider("Height",50,h-top,min(300,h-top))
    cropped = img.crop((left,top,left+crop_w,top+crop_h))
    st.image(cropped, caption="Preview", use_column_width=True)
    return cropped

# ---------------- Main ----------------
if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Original")
    st.image(original, use_column_width=True)

    # AI passport photo
    st.header("AI Passport Photo (630x810 px)")
    face = detect_face(original)
    if face is None:
        st.error("Face not detected. Use manual crop.")
    else:
        beard_flag = (photo_type=="With Beard")
        cropped = ai_crop(original, face, subject_type, beard_flag)
        bg_removed = grabcut_bg(cropped, face)
        enhanced = enhance(bg_removed)
        final_ai = enhanced.resize((630,810), Image.LANCZOS)
        final_ai = add_border(final_ai)
        st.image(final_ai, caption="AI Passport Photo")
        buf = io.BytesIO()
        final_ai.save(buf,"JPEG",quality=95)
        st.download_button("Download AI Passport Photo (630x810)", buf.getvalue(), file_name="passport_ai_630x810.jpg", mime="image/jpeg")

    # Manual crop
    st.header("Manual Crop Tool")
    manual = manual_crop(original)
    if manual:
        manual_resized = manual.resize((630,810), Image.LANCZOS)
        buf2 = io.BytesIO(); manual_resized.save(buf2,"JPEG",quality=95)
        st.download_button("Download Manual Crop (630x810)", buf2.getvalue(), file_name="manual_630x810.jpg", mime="image/jpeg")
