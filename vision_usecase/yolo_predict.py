import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect people in an image using the local YOLO weights."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SCRIPT_DIR / "input.jpg",
        help="Input image path (default: input.jpg in this folder).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=SCRIPT_DIR / "yolov8n.pt",
        help="YOLO weights path (default: yolov8n.pt in this folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "yolo_person_detection.jpg",
        help="Path for the annotated output image.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1.")

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.image),
        classes=[0],
        conf=args.confidence,
        verbose=False,
    )

    result = results[0]
    annotated_image = result.plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.output), annotated_image):
        raise OSError(f"Could not save detection result to: {args.output}")

    print(f"People detected: {len(result.boxes)}")
    print(f"Detection result saved to: {args.output}")


if __name__ == "__main__":
    main()
