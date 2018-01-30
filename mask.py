import model as modellib
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
    model = None

    def init(self):
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

    def detect_people(self, image):
        results = model.detect([image])
        r = results[0]
