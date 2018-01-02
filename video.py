import imageio
import detect
import PIL.Image as Image
import numpy as np
import sys
import time

start = time.time()

filename = sys.argv[1]
if not filename:
    raise Exception("No filename included")
    sys.exit()

reader = imageio.get_reader(filename, "ffmpeg")
fps = reader.get_meta_data()['fps']
N = len(reader) - 1

writer = imageio.get_writer("output_" + filename, fps=fps)

for i, frame in enumerate(reader):
    print("\r{}/{}".format(i, N), end="")
    new_frame = detect.detect_image(
            Image.fromarray(np.uint8(frame)).convert('RGB'),
            use_same_colour = False
        )
    writer.append_data(new_frame)

writer.close()
end = time.time()

print("Execution time: {}s".format(end-start))
