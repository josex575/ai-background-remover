# ---- CROP + ALIGN FUNCTION ----
def crop_head_only(image: Image.Image, face_box):
    """
    Crops the image to include head and a little neck, no shoulders.
    Leaves small space above head and centers the head.
    """
    x, y, w, h = face_box
    np_img = np.array(image)
    img_h, img_w = np_img.shape[:2]

    # Top margin (~2cm in pixels assuming 300dpi)
    dpi = 300
    cm2px = dpi / 2.54
    top_margin = int(2 * cm2px)

    # Bottom extension (~10% of face height for small neck)
    bottom_margin = int(h * 0.1)

    # Horizontal margins (slightly wider than face)
    x1 = max(x - w // 8, 0)
    x2 = min(x + w + w // 8, img_w)

    # Vertical margins
    y1 = max(y - top_margin, 0)
    y2 = min(y + h + bottom_margin, img_h)

    cropped = image.crop((x1, y1, x2, y2))

    # Resize to passport ratio 4:5
    target_ratio = 4 / 5
    cropped_w, cropped_h = cropped.size
    current_ratio = cropped_w / cropped_h

    if current_ratio > target_ratio:
        # too wide → pad height
        new_h = int(cropped_w / target_ratio)
        pad_h = new_h - cropped_h
        cropped = ImageOps.expand(cropped, border=(0, pad_h // 2), fill="white")
    elif current_ratio < target_ratio:
        # too tall → pad width
        new_w = int(cropped_h * target_ratio)
        pad_w = new_w - cropped_w
        cropped = ImageOps.expand(cropped, border=(pad_w // 2, 0), fill="white")

    return cropped
