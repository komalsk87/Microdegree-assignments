import cv2
import numpy as np
from ultralytics import YOLO


def apply_80s_retro_filter(image):
    """Applies an 80s vintage filter: warm color shift, high contrast,

    increased saturation, soft bloom, and film grain.
    """
    img_float = image.astype(np.float32) / 255.0

    # 1. Warm Vintage Color Grading (Boost Red, soft Green, cut Blue)
    b, g, r = cv2.split(img_float)
    r = np.clip(r * 1.25 + 0.05, 0, 1)  # Warm reds/pinks
    g = np.clip(g * 1.05 + 0.02, 0, 1)  # Slight green boost
    b = np.clip(b * 0.85 - 0.02, 0, 1)  # Reduce cold blues
    vintage = cv2.merge([b, g, r])

    # 2. Convert to HSV to boost Saturation & Contrast
    vintage_uint8 = (vintage * 255).astype(np.uint8)
    hsv = cv2.cvtColor(vintage_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)  # Saturate neon colors
    vintage_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Soft Glow / Bloom Effect
    blur = cv2.GaussianBlur(vintage_uint8, (21, 21), 0)
    bloomed = cv2.addWeighted(vintage_uint8, 0.75, blur, 0.35, 0)

    # 4. Add 80s Analog Film Grain
    h, w, c = image.shape
    noise = np.random.normal(0, 12, (h, w, c)).astype(np.float32)
    retro_final = np.clip(bloomed.astype(np.float32) + noise, 0, 255).astype(
        np.uint8
    )

    return retro_final


def process_80s_trend(image_path, output_path="output_80s.jpg"):
    # Load YOLOv8 model (downloads lightweight model automatically)
    model = YOLO("yolov8n.pt")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Run YOLO detection
    results = model(image)

    # Process detections
    processed_image = image.copy()

    for result in results:
        for box in result.boxes:
            # Check if detected object is a person (Class ID 0 in COCO dataset)
            if int(box.cls[0]) == 0:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract person crop
                person_crop = image[y1:y2, x1:x2]

                if person_crop.size > 0:
                    # Apply retro filter on the person
                    filtered_person = apply_80s_retro_filter(person_crop)

                    # Smooth blend around edges (optional feathering)
                    processed_image[y1:y2, x1:x2] = filtered_person

    # Save and display result
    cv2.imwrite(output_path, processed_image)
    print(f"80s Trend photo saved to {output_path}")


# Run the function on your image
process_80s_trend("input.jpg", "output_80s.jpg")