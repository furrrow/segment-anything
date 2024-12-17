from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import torch
import supervision as sv
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pprint import pprint
import tyro
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

@dataclass
class Config:
    # path for segment anything's saved weights
    checkpoint_path: str = "../sam_vit_h_4b8939.pth"

    # folder where the images that should be processed live
    image_folder: str = "/media/jim/Hard Disk/GND/UMD/0813_trail/camera_processed"

    # output folder name
    output_folder: str = "../annotation/"

    # model type, a parameter related to segment anything model init
    model_type: str = "vit_h"

    # Pass --no-cuda in to use cpu only.
    cuda: bool = True

    # which image to start processing from, allows skip if some processing is already done.
    start_idx: int = 0

    # saves individual segmented masks as its own img, pass --no-save_sam_masks in to set this value to False.
    save_sam_masks: bool = True

    # saves individual traversability masks as its own image, pass --no-save_sam_masks in to set this value to False.
    save_traverse_masks: bool = True


def main(config: Config):
    FIGSIZE = (10, 8)
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

    for idx, image_name in enumerate(os.listdir(config.image_folder)):
        if idx < config.start_idx:
            continue
        image_path = os.path.join(config.image_folder, image_name)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        save_folder = os.path.join(output_folder, image_name[:-4])
        if os.path.exists(save_folder):
            print(f"path {save_folder} already exists, skipping...")
            continue
        os.makedirs(save_folder)
        metadata_file = open(f"{save_folder}/meta.json", "w")

        masks = mask_generator.generate(image_rgb)
        empty_mask = {"segmentation": np.zeros(masks[0]['segmentation'].shape).astype(bool),
                      'area': 0,
                      'bbox': [0, 0, 1, 1]}
        traverse_masks = [empty_mask.copy(), empty_mask.copy(), empty_mask.copy(), empty_mask.copy(), empty_mask.copy()]
        image_np = np.array(image_bgr)
        # print("image shape", image_np.shape)
        # print(f"masks length: {len(masks)}")
        # print(f"fields in each mask: {masks[0].keys()}")
        # print(f"segmentation mask shape: {masks[0]['segmentation'].shape}")

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        traverse_dict = {
            1: "1. side walk or other traversable areas for pedestrians",
            2: "2. the parking lot, drive way for vehicles",
            3: "3. off-road including vegetation, grass, mud etc",
            4: "4. stairs, curbs etc for legged robot only",
            5: "5. obstacles or buildings",
            6: "6. do not segment, discard"
        }
        prompt_text = (f"{traverse_dict[1]}; \n"
                       f"{traverse_dict[2]}; \n"
                       f"{traverse_dict[3]}; \n"
                       f"{traverse_dict[4]}; \n"
                       f"{traverse_dict[5]}; \n"
                       f"{traverse_dict[6]};")
        for i, i_mask in enumerate(masks):
            detections = sv.Detections.from_sam([i_mask])
            annotated_frame = mask_annotator.annotate(
                scene=image_bgr.copy(),
                detections=detections
            )
            # plot just the annotation
            plt.figure(figsize=FIGSIZE)
            plt.imshow(annotated_frame)
            # plt.axis("off")
            print("====")
            print(f"Showing Segmentation Mask {i} out of {len(masks)}")
            print(f"{prompt_text}\n press q to proceed: ")
            if config.save_sam_masks: plt.savefig(f"{save_folder}/sam_{i}.png")
            plt.show()

            # Prompting the user to enter numbers
            while True:
                user_mask = input(f"Please Enter Traversability Category 1 through 6: ")
                if user_mask == "":
                    traverse_category = 5
                    print(f"you pressed enter, defaulting to {traverse_dict[traverse_category]}")
                    break
                elif 1 <= int(user_mask) <= 6:
                    traverse_category = int(user_mask)
                    print(f"you entered {traverse_dict[traverse_category]}")
                    break
                else:
                    print("input invalid, please enter an int between 1 and 6")
            if traverse_category == 6:
                # ignore this category
                continue
            # overlay mask with the right traverse category in traverse_masks
            traverse_masks[traverse_category - 1]['segmentation'] = np.logical_or(
                traverse_masks[traverse_category - 1]['segmentation'], i_mask['segmentation'])
            traverse_masks[traverse_category - 1]['area'] = np.sum(
                traverse_masks[traverse_category - 1]['segmentation'])
            metadict = masks[i].copy()
            metadict.pop('segmentation', None)
            metadict['traverse_mask'] = traverse_category
            metadata_file.write('\"{}\": {},\n'.format(i, json.dumps(metadict)))
        metadata_file.close()

        # save individual category's segmentations:
        for i, i_mask in enumerate(traverse_masks):
            detections = sv.Detections.from_sam([i_mask])
            annotated_frame = mask_annotator.annotate(
                scene=image_bgr.copy(),
                detections=detections
            )
            # plot just the annotation
            implot = plt.imshow(annotated_frame)
            # plt.axis("off")
            if config.save_traverse_masks: plt.savefig(f"{save_folder}/traverse_{i + 1}.png")

        # show final traversability annotation:
        detections = sv.Detections.from_sam(traverse_masks)
        annotated_frame = mask_annotator.annotate(
            scene=image_bgr.copy(),
            detections=detections
        )
        plt.imshow(annotated_frame)
        # plt.axis("off")
        plt.savefig(f"{save_folder}/traversability_mask.png")
        plt.show()
        mask_array = np.array([mask['segmentation'] for mask in traverse_masks])
        np.save(f"{save_folder}/traversability_mask.npy", mask_array)
        print(f"saved to {save_folder}")


if __name__ == "__main__":
    """
    this script uses tyro to enable command line arguments. 
    for example: python ./scripts/segment_bulk.py --help
    """
    config = tyro.cli(Config)
    pprint(config)
    main(config)
