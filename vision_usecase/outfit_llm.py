import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from ultralytics import YOLO


def change_clothes_to_80s(
    image_path, output_path="80s_outfit.png", prompt="80s retro outfit"
):
    # 1. Load YOLO Segmentation model (yolov8n-seg.pt detects precise shapes/polygons)
    yolo_model = YOLO("yolov8n-seg.pt")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")

    h, w, _ = image.shape
    results = yolo_model(image)

    # Create a blank mask for the clothing area
    mask = np.zeros((h, w), dtype=np.uint8)

    for result in results:
        if result.masks is not None:
            for mask_data, cls_id in zip(
                result.masks.data, result.boxes.cls
            ):
                # Class 0 is person
                if int(cls_id) == 0:
                    # Convert tensor mask to numpy array
                    person_mask = (
                        mask_data.cpu().numpy().astype(np.uint8) * 255
                    )
                    person_mask = cv2.resize(person_mask, (w, h))

                    # Crop mask to target mostly torso/body area (rough clothing mask)
                    # For a precise clothing-only mask, you can use specialized models like fashion-clip
                    mask = cv2.bitwise_or(mask, person_mask)

    # Save mask to PIL Image format for Stable Diffusion
    mask_pil = Image.fromarray(mask)
    init_image_pil = Image.open(image_path).convert("RGB")

    # 2. Load Stable Diffusion Inpainting Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Diffusion model on: {device}...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    # Prompt describing 80s clothing fashion
    full_prompt = f"a person wearing {prompt}, vintage 1980s neon synthwave jacket, denim, stylish retro pattern, highly detailed 80s aesthetic photo"
    negative_prompt = "modern clothing, suit, t-shirt, simple modern dress, low quality, blurry"

    # Run AI Inpainting to regenerate clothes region
    print("Generating 80s clothing...")
    result_image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        image=init_image_pil,
        mask_image=mask_pil,
    ).images[0]

    result_image.save(output_path)
    print(f"Done! Saved 80s outfit photo to {output_path}")


# Run the function
change_clothes_to_80s(
    "input.jpg",
    "output_80s_outfit.png",
    prompt="vintage 80s neon tracksuit and denim jacket",
)