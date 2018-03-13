import imageio
import PIL.Image as Image
import numpy as np
import model as modellib
import coco
from mask import MaskRCNN
from objects import Frames
import os
import time
from tqdm import *

begin = time.time()

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

#mask_rcnn = MaskRCNN()
#output_frames = mask_rcnn.detect_people_multiframes(frames, BATCH_SIZE = BATCH_SIZE)

output_frames = []
max_batch_no = int(len(frames)/BATCH_SIZE)
for i in tqdm(range(max_batch_no), desc="Detecting" ):
    now = time.time()
    frames_batch = frames.get_batch(i)

    batch_length = len(frames_batch)
    config.BATCH_SIZE = batch_length
    config.IMAGES_PER_GPU = batch_length

    output = model.detect(frames_batch) #works


    for output_frame in output:
        output_frames.append(output_frame)

writer = imageio.get_writer("output_test.mkv", fps = fps)
for frame in tqdm(output_frames, desc = "Writing to file"):
    try:
        writer.append_data(np.array(frame))
    except:
        pass

writer.close()
end = time.time()
print("Total time elapsed: {}s".format(int(end - begin)))

