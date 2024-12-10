from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import torch
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
"""
https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
https://www.kaggle.com/code/mrinalmathur/segment-anything-model-tutorial
"""

def show_anns(anns, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m**0.5)))


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "../sam_vit_h_4b8939.pth"
IMAGE_PATH = "../images/pronav-cover.jpg"
# IMAGE_PATH = "../images/spot_outdoor.png"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image_rgb)

image_np = np.array(image_bgr)
# Reshape the image data to a valid shape
print("image shape", image_np.shape)
print(f"masks length: {len(masks)}")
print(f"fields in each mask: {masks[0].keys()}")
print(f"segmentation mask shape: {masks[0]['segmentation'].shape}")

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(masks)
annotated_frame = mask_annotator.annotate(
    scene=image_bgr.copy(),
    detections=detections
)

# If we want to plot the original image and the mask
# fig, axs = plt.subplots(1, 2, figsize=(16, 16))
# axs[0].imshow(image_rgb)
# axs[1].imshow(annotated_frame)
# axs[0].axis('off')
# axs[1].axis('off')

# plot just the annotation
implot = plt.imshow(annotated_frame)
plt.axis("off")
# plt.show()
plt.savefig("figure.png")