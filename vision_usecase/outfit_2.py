import cv2
import numpy as np
from ultralytics import YOLO


def overlay_png(bg, fg, x, y, w, h):
    """Overlays a 4-channel PNG (with alpha transparency) onto a 3-channel background image."""
    # Resize foreground dress PNG to target bounding box
    fg_resized = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)

    # Extract RGB and Alpha channels
    fg_rgb = fg_resized[:, :, :3]
    alpha = fg_resized[:, :, 3] / 255.0

    # Ensure coordinates stay within frame bounds
    h_bg, w_bg, _ = bg.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w), min(h_bg, y + h)

    # Crop overlay and alpha if partially outside canvas
    fg_crop = fg_rgb[0 : (y2 - y1), 0 : (x2 - x1)]
    alpha_crop = alpha[0 : (y2 - y1), 0 : (x2 - x1)]

    # Blend alpha channel
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            alpha_crop * fg_crop[:, :, c] + (1 - alpha_crop) * bg[y1:y2, x1:x2, c]
        )

    return bg


def swap_to_retro_dress(
    image_path,
    dress_path="80s_jacket.png",
    bg_path="retro_bg.jpg",
    output_path="80s_final.jpg",
):
    # Load YOLO Pose model to accurately find shoulders/torso anchor points
    model = YOLO("yolov8n-pose.pt")

    image = cv2.imread(image_path)
    dress = cv2.imread(dress_path, cv2.IMREAD_UNCHANGED)

    if image is None or dress is None:
        raise FileNotFoundError(
            "Check that input.jpg and 80s_jacket.png exist in your folder."
        )

    # Convert 3-channel dress to 4-channel if it lacks transparency
    if dress.shape[2] == 3:
        b, g, r = cv2.split(dress)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        dress = cv2.merge([b, g, r, alpha])

    results = model(image)
    output = image.copy()

    for result in results:
        # Check detected person keypoints (shoulders/hips)
        if result.keypoints is not None and len(result.keypoints) > 0:
            for kpts in result.keypoints.data:
                # Keypoint index 5 = Left Shoulder, Keypoint 6 = Right Shoulder
                l_shoulder = kpts[5][:2].cpu().numpy()
                r_shoulder = kpts[6][:2].cpu().numpy()

                if l_shoulder[0] > 0 and r_shoulder[0] > 0:
                    # Calculate shoulder distance to automatically scale retro dress
                    shoulder_width = int(
                        np.linalg_norm(l_shoulder - r_shoulder)
                    )

                    # Estimate clothing anchor width and height based on shoulders
                    dress_w = int(shoulder_width * 2.2)
                    dress_h = int(dress_w * (dress.shape[0] / dress.shape[1]))

                    # Calculate alignment position over person's torso
                    center_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                    top_y = int(
                        min(l_shoulder[1], r_shoulder[1]) - (dress_h * 0.15)
                    )
                    top_x = center_x - (dress_w // 2)

                    # Overlay retro outfit
                    output = overlay_png(
                        output, dress, top_x, top_y, dress_w, dress_h
                    )

    # Apply soft 80s film color grade to finalize look
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)  # Boost neon colors
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, graded)
    print(f"Retro photo successfully created: {output_path}")


# Execute transformation
swap_to_retro_dress("input.jpg", "80s_jacket.png")