from PIL import Image
import model as modellib
import numpy as np
import os
import utils
import coco

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

class Mask(object):
    class_id = None
    score = None
    mask = None
    rois = None
    masked_image = None
    upper_half = None


    def __init__(self, class_id, mask, rois, score, masked_image):
        self.class_id = class_id
        self.mask = mask
        self.rois = rois
        self.score = score
        self.masked_image = masked_image

        width, height = masked_image.size
        self.upper_half = np.array(masked_image.crop((0, 0, width, int(height/2))))


class Masks(list):

    def __get__(self):
        return


class MaskRCNN(object):
    model = None

    def __init__(self):
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def detect_people(self, image):
        results = self.model.detect([image])
        r = results[0]

        masks = []
        for i, class_id in enumerate(r["class_ids"]):
            if class_id != 1:
                continue
            crop = r["rois"][i]
            score = r["scores"][i]
            this_mask = r["masks"][:, :, i:i+1]
            subimage = np.multiply(this_mask,image)

            mask = Mask(
                class_id,
                this_mask,
                crop,
                score,
                Image.fromarray(subimage).crop((crop[1], crop[0], crop[3], crop[2]))
            )
            masks.append(mask)
        return Masks(masks)

