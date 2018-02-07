from PIL import Image
from operator import itemgetter
import model as modellib
import numpy as np
import os
import math
import utils
import coco

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class Mask(object):
    class_id = None
    score = None
    mask = None
    rois = None
    masked_image = None
    masked_image_np = None
    upper_half = None
    upper_half_np = None
    colour = None
    boundary_index = None
    center = None


    def __init__(self, class_id, mask, rois, score, masked_image):
        self.class_id = class_id
        self.mask = mask
        self.rois = rois
        self.center = (
            int((rois[3] + rois[1])/2),
            int((rois[2] + rois[0])/2)
        )
        self.score = score
        self.masked_image = masked_image
        self.masked_image_np = np.array(masked_image)

        width, height = masked_image.size
        self.upper_half = masked_image.crop((0, 0, width, int(height/2)))
        self.upper_half_np = np.array(self.upper_half)


class Masks(list):

    def filter_boundary_index(self, index):
        result = []
        for mask in self:
            if mask.boundary_index == index:
                result.append(mask)
        return Masks(result)

    def distances(self):
        n = len(self)
        result = []

        for i in range(n):
            p1 = self[i].center
            for j in range(i+1, n):
                p2 = self[j].center
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                result.append([i, j, distance])
        result.sort(key=itemgetter(2))
        return np.array(result)





class MaskRCNN(object):
    model = None
    config = None

    def __init__(self):
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def detect_people(self, image):
        results = self.model.detect([image])
        r = results[0]

        self.config.BATCH_SIZE = 1
        self.config.IMAGES_PER_GPU = 1

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


    def detect_people_multiframes(self, images):
        no_of_images = len(images)
        self.config.BATCH_SIZE = no_of_images
        self.config.IMAGES_PER_GPU = no_of_images

        results = self.model.detect(images)
        frame_masks = []
        for r in results:
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
            frame_masks.append(Masks(masks))

        return frame_masks


