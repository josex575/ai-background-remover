import streamlit as st
from rembg import remove
from PIL import Image
import numpy as np
import io

st.set_page_config(page_title="AI Background Remover & Cropper", layout="centered")

st.title("🪄 AI Background Remover & Cropper")
st.markdown("Upload an image, and let AI automatically remove the background and crop the subject.")

uploaded = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

def auto_crop(image: Image.Image):
    """Automatically crops to the non-transparent area of an image."""
    np_img = np.array(image)
    if np_img.shape[2] < 4:
        return image  # No alpha channel found
    alpha = np_img[:, :, 3]
    coords = np.argwhere(alpha > 0)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)[:2]
    y1, x1 = coords.max(axis=0)[:2] + 1
    return image.crop((x0, y0, x1, y1))

if uploaded:
    # Display original
    image = Image.open(uploaded)
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("🪄 Removing background..."):
        output = remove(image)
        cropped = auto_crop(output)

    st.image(output, caption="Background Removed", use_container_width=True)
    st.image(cropped, caption="Cropped to Subject", use_container_width=True)

    # Prepare cropped image for download
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Download Cropped Image",
        data=byte_im,
        file_name="cropped.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image file to get started!")
