import torch
import torch.nn as nn


class PersonDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------
        # CNN Feature Extractor
        # --------------------------------

        self.backbone = nn.Sequential(

            # Layer 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # Convert feature map
            # to 1 x 1 feature vector
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # --------------------------------
        # Classification
        # --------------------------------

        self.classifier = nn.Linear(
            256,
            2
        )

        # 0 = background
        # 1 = person

        # --------------------------------
        # Bounding box regression
        # --------------------------------

        self.box_regressor = nn.Linear(
            256,
            4
        )

    def forward(self, x):

        # CNN
        features = self.backbone(x)

        # Flatten
        features = features.flatten(1)

        # Classification
        class_logits = self.classifier(
            features
        )

        # Bounding box
        boxes = self.box_regressor(
            features
        )

        return class_logits, boxes