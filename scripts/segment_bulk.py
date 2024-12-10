from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import torch
import supervision as sv
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tyro
from tqdm import tqdm

@dataclass
class Config:
    checkpoint_path: str = "../sam_vit_h_4b8939.pth"
    image_folder: str = "/media/jim/Hard Disk/GND/UMD/0813_trail/camera_processed"
    output_folder: str = "annotation/"
    model_type: str = "vit_h"
    cuda: bool = True



if __name__ == "__main__":
    config = tyro.cli(Config)
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and config.cuda) else 'cpu')
    sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint_path)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)

    if os.path.isdir(config.image_folder):
        print(f"extracting from {config.image_folder}")
        output_folder = os.path.join(config.image_folder, config.output_folder)
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    else:
        print(f"{config.image_folder} is not a directory.")
        exit()

    for image_name in tqdm(os.listdir(config.image_folder)):
        image_path = os.path.join(config.image_folder, image_name)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(masks)
        annotated_frame = mask_annotator.annotate(
            scene=image_bgr.copy(),
            detections=detections
        )
        plt.imshow(annotated_frame)
        save_name = f"{image_name}_annotated.png"
        save_path = os.path.join(output_folder, save_name)
        plt.savefig(save_path)

    print ("saved to", output_folder)
