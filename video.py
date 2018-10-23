import imageio
import detect
import utils
import PIL.Image as Image
import numpy as np
import sys
import time
from tqdm import *

print("Start")

start = time.time()

filename = sys.argv[1]
if not filename:
    raise Exception("No filename included")
    sys.exit()

reader = imageio.get_reader(filename, "ffmpeg")
fps = reader.get_meta_data()['fps']
N = len(reader) - 1

writer = imageio.get_writer("output/" + filename, fps=fps)

new_frames = detect.detect_images(reader, BATCH_SIZE = 16, threshold = 0.9)
for frame in tqdm(new_frames, "Writing frames"):
    writer.append_data(frame)

writer.close()
end = time.time()

print("\n\nExecution time: {}s".format(end-start))
