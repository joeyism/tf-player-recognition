
import imageio
import PIL.Image as Image
import numpy as np
import model as modellib
import coco
from objects import Frames
import os

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()
config.display()


reader = imageio.get_reader("firmino-arsenal-2.mkv", "ffmpeg")
fps = reader.get_meta_data()["fps"]
frames = np.array([ np.uint8(frame) for i, frame in enumerate(reader)])
BATCH_SIZE = 16


config.BATCH_SIZE = BATCH_SIZE
config.IMAGES_PER_GPU = BATCH_SIZE

ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
frames = Frames(frames)
print("BATCH SIZE: ", config.BATCH_SIZE)
for i in range(int(len(frames)/BATCH_SIZE)):
    PRINT("\r{}/{}\t".format(i, int(len(frames)/BATCH_SIZE)), end="")
    frames_batch = frames.get_batch(i)
    model.detect(frames_batch)
