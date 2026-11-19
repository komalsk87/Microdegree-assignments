import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PersonDataset(Dataset):

    def __init__(self, root):

        self.root = root

        self.image_dir = os.path.join(
            root, "PNGImages"
        )

        self.mask_dir = os.path.join(
            root, "PedMasks"
        )

        self.images = sorted(
            os.listdir(self.image_dir)
        )

        self.masks = sorted(
            os.listdir(self.mask_dir)
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        # -----------------------------
        # Image path
        # -----------------------------

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        # -----------------------------
        # Load image
        # -----------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # -----------------------------
        # Load mask
        # -----------------------------

        mask = np.array(
            Image.open(mask_path)
        )

        # -----------------------------
        # Find individual persons
        # -----------------------------

        person_ids = np.unique(mask)

        # 0 = background
        person_ids = person_ids[
            person_ids != 0
        ]

        boxes = []

        for person_id in person_ids:

            ys, xs = np.where(
                mask == person_id
            )

            xmin = xs.min()
            ymin = ys.min()

            xmax = xs.max()
            ymax = ys.max()

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        # -----------------------------
        # Labels
        # -----------------------------

        # 1 = person
        labels = torch.ones(
            len(boxes),
            dtype=torch.int64
        )
        

        # -----------------------------
        # Convert image to tensor
        # -----------------------------

        image = TF.to_tensor(image)

        # -----------------------------
        # Target dictionary
        # -----------------------------

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target


# --------------------------------------
# Important for object detection
# --------------------------------------

def collate_fn(batch):

    return tuple(zip(*batch))
