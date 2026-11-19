import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import PersonDataset, collate_fn
from model_building import PersonDetector


# ==========================================
# 1. Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# ==========================================
# 2. Dataset
# ==========================================

dataset = PersonDataset(
    "data/PennFudanPed"
)

print(
    "Number of images:",
    len(dataset)
)


# ==========================================
# 3. DataLoader
# ==========================================

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)


# ==========================================
# 4. Model
# ==========================================

model = PersonDetector()

model = model.to(device)


# ==========================================
# 5. Loss functions
# ==========================================

classification_loss = nn.CrossEntropyLoss()

box_loss = nn.SmoothL1Loss()


# ==========================================
# 6. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. Training
# ==========================================

epochs = 1000


for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, targets in loader:

        # ----------------------------------
        # Because batch_size = 1
        # ----------------------------------

        image = images[0]

        target = targets[0]

        # ----------------------------------
        # Move image to device
        # ----------------------------------

        image = image.unsqueeze(0)

        image = image.to(device)

        # ----------------------------------
        # Get target boxes
        # ----------------------------------

        target_boxes = target["boxes"]

        # ----------------------------------
        # IMPORTANT:
        # Our simple model predicts
        # only ONE bounding box.
        #
        # Therefore use the first person.
        # ----------------------------------

        target_box = target_boxes[0]

        target_box = target_box.to(device)

        # ----------------------------------
        # Target label
        # ----------------------------------

        target_label = torch.tensor(
            [1],
            dtype=torch.long,
            device=device
        )

        # ----------------------------------
        # Forward pass
        # ----------------------------------

        class_logits, predicted_box = model(
            image
        )

        # ----------------------------------
        # Classification loss
        # ----------------------------------

        loss_cls = classification_loss(
            class_logits,
            target_label
        )

        # ----------------------------------
        # Bounding box loss
        # ----------------------------------

        loss_box = box_loss(
            predicted_box[0],
            target_box
        )

        # ----------------------------------
        # Total loss
        # ----------------------------------

        loss = loss_cls + loss_box

        # ----------------------------------
        # Backpropagation
        # ----------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # --------------------------------------
    # Epoch result
    # --------------------------------------

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# 8. Save model
# ==========================================

torch.save(
    model.state_dict(),
    "person_detector.pth"
)

print()
print("Training completed!")
print("Model saved as person_detector.pth")