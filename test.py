
import imageio
import PIL.Image as Image
import numpy as np
import model as modellib
import coco
from objects import Frames
import os
import time

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()


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
frames.BATCH_SIZE = BATCH_SIZE


config.display()

print("BATCH SIZE: ", config.BATCH_SIZE)
begin = time.time()
for i in range(int(len(frames)/BATCH_SIZE)):
    now = time.time()
    print("\r{}/{} {}s\t".format(i + 1, int(len(frames)/BATCH_SIZE), int(now - begin)), end="")
    frames_batch = frames.get_batch(i)
    model.detect(frames_batch)
end = time.time()
print("Total time elapsed: {}s".format(int(end - begin)))

