import torch
import cv2
import numpy as np

from model_building import PersonDetector


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# 2. Load model
# ==========================================

model = PersonDetector()

model.load_state_dict(
    torch.load(
        "person_detector.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ==========================================
# 3. Load image
# ==========================================

image = cv2.imread(
    "test_1.jpg"
)

if image is None:

    raise FileNotFoundError(
        "test_1.jpg not found"
    )


# ==========================================
# 4. Convert BGR → RGB
# ==========================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 5. Convert image to tensor
# ==========================================

tensor = torch.from_numpy(
    image_rgb
).float()

tensor = tensor / 255.0

# H,W,C → C,H,W

tensor = tensor.permute(
    2,
    0,
    1
)

# Add batch dimension

tensor = tensor.unsqueeze(0)

tensor = tensor.to(device)


# ==========================================
# 6. Prediction
# ==========================================

with torch.no_grad():

    class_logits, predicted_box = model(
        tensor
    )

    # --------------------------------------
    # Class probabilities
    # --------------------------------------

    probabilities = torch.softmax(
        class_logits,
        dim=1
    )

    person_confidence = probabilities[
        0,
        1
    ].item()

    # --------------------------------------
    # Bounding box
    # --------------------------------------

    box = predicted_box[
        0
    ].cpu().numpy()


# ==========================================
# 7. Print results
# ==========================================

print(
    "Person confidence:",
    person_confidence
)

print(
    "Predicted box:",
    box
)


# ==========================================
# 8. Draw bounding box
# ==========================================

if person_confidence > 0.5:

    x1, y1, x2, y2 = box

    # Convert to integers

    x1 = int(x1)
    y1 = int(y1)

    x2 = int(x2)
    y2 = int(y2)

    # --------------------------------------
    # Keep coordinates inside image
    # --------------------------------------

    h, w = image.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # --------------------------------------
    # Draw rectangle
    # --------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # --------------------------------------
    # Label
    # --------------------------------------

    text = (
        f"Person "
        f"{person_confidence:.2f}"
    )

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ==========================================
# 9. Save result
# ==========================================

cv2.imwrite(
    "result.jpg",
    image
)

print(
    "Detection result saved as result.jpg"
)