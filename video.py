import imageio
import detect
import PIL.Image as Image
import numpy as np
import sys

filename = sys.argv[1]
if not filename:
    raise Exception("No filename included")
    sys.exit()

reader = imageio.get_reader(filename, "ffmpeg")
fps = reader.get_meta_data()['fps']
N = len(reader)

writer = imageio.get_writer("output_" + filename, fps=fps)

try:
    for i, frame in enumerate(reader):
        print("\r{}/{}".format(i, N), end="")
        new_frame = detect.detect_image(Image.fromarray(np.uint8(frame)).convert('RGB')
    )
        writer.append_data(new_frame)
except:
    pass

writer.close()
