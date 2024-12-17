from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import torch
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
"""
https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
https://www.kaggle.com/code/mrinalmathur/segment-anything-model-tutorial
"""

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "../sam_vit_h_4b8939.pth"
# IMAGE_PATH = "../images/spot_outdoor.png"
IMAGE_PATH = "../images/UMD_01.png"
save_path = "./saves/"
metadata_file = open(f"{save_path}/meta.json", "w")
SAVE_SAM_MASKS = True  # saves individual masks from segment-anything
SAVE_TRAVERSE_MASKS = True # saves individual category masks from 5 categories
FIGSIZE = (10, 8)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam,
                                           pred_iou_thresh=0.90,
                                           stability_score_thresh=0.96)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image_rgb)
empty_mask = {"segmentation": np.zeros(masks[0]['segmentation'].shape).astype(bool),
              'area': 0,
              'bbox': [0, 0, 1, 1]}
traverse_masks = [empty_mask.copy(), empty_mask.copy(), empty_mask.copy(), empty_mask.copy(), empty_mask.copy()]
image_np = np.array(image_bgr)
print("image shape", image_np.shape)
print(f"masks length: {len(masks)}")
print(f"fields in each mask: {masks[0].keys()}")
print(f"segmentation mask shape: {masks[0]['segmentation'].shape}")

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
    plt.figure(figsize=(FIGSIZE))
    implot = plt.imshow(annotated_frame)
    # plt.axis("off")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print("====")
    print(f"Showing Segmentation Mask {i} out of {len(masks)}")
    print(f"{prompt_text}\n press q to proceed: ")
    if SAVE_SAM_MASKS: plt.savefig(f"{save_path}/sam_{i}.png")
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
    traverse_masks[traverse_category - 1]['segmentation'] = np.logical_or(traverse_masks[traverse_category-1]['segmentation'], i_mask['segmentation'])
    traverse_masks[traverse_category - 1]['area'] = np.sum(traverse_masks[traverse_category-1]['segmentation'])
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
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if SAVE_TRAVERSE_MASKS: plt.savefig(f"{save_path}/traverse_{i+1}.png")

# show final traversability annotation:
detections = sv.Detections.from_sam(traverse_masks)
annotated_frame = mask_annotator.annotate(
    scene=image_bgr.copy(),
    detections=detections
)
implot = plt.imshow(annotated_frame)
# plt.axis("off")
plt.savefig(f"{save_path}/traversability_mask.png")
plt.show()
mask_array = np.array([mask['segmentation'] for mask in traverse_masks])
np.save("traversability_mask.npy", mask_array)