from PIL import Image, ImageEnhance
from objects import Detections
from functools import partial
from timeme import time_here
from mask import MaskRCNN
import tensorflow as tf
import numpy as np
import os
import label_map_util
import utils
import sys
import multiprocessing as mp
import time

OUTPUT_FOLDER = "output"

mask_rcnn = MaskRCNN()

def detect_images(images, use_same_colour = True, BATCH_SIZE = 16):
    print("Detect images")
    frames = []
    try:
        for image in images:
            frames.append(utils.load_image_into_numpy_array(image))
    except:
        pass
    print("There are {} frames".format(len(frames)))

    frame_masks = mask_rcnn.detect_people_multiframes(frames, BATCH_SIZE = BATCH_SIZE)

    print("Complete detection")

    N = len(frame_masks)
    for i, masks in enumerate(frame_masks):
        print("\r{}/{}, {time}s\t".format(i, N, time=time.time() - start), end="")
        for mask in masks:
            mask.colour = utils.get_colours_from_image(mask.upper_half_np)
            mask.boundary_index = int(utils.get_boundary_num(mask.colour))

        utils.draw_ellipses_around_masks(frame[i], masks)
        utils.draw_lines_between_players(frame[i], masks)

    return frames

def detect_image(image, detection_threshold = 0.4, colour_threshold = 20, use_same_colour = True):
    old_image = image.copy()
    old_image_np = utils.load_image_into_numpy_array(old_image)
    masks = mask_rcnn.detect_people(old_image_np)
    for mask in masks:
        mask.colour = utils.get_colours_from_image(mask.upper_half_np)
        mask.boundary_index = int(utils.get_boundary_num(mask.colour))

    utils.draw_ellipses_around_masks(old_image_np, masks)
    utils.draw_lines_between_players(old_image_np, masks)

    return old_image_np



def test():
    TEST_IMAGES_FOLDER = "sports_images"
    TEST_IMAGES_PATHS = [ os.path.join(TEST_IMAGES_FOLDER, image_path) for image_path in os.listdir(TEST_IMAGES_FOLDER)]
    for i, image_path in enumerate(TEST_IMAGES_PATHS):
        time_here("{} {}".format(i, image_path))
        image = Image.open(image_path)
        image_np = detect_image(image, use_same_colour = False)
        Image.fromarray(image_np).save(OUTPUT_FOLDER + "/" + image_path.replace("/", "_"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
        sys.exit()
    path = sys.argv[1]
    image = Image.open(path)
    image_np = detect_image(image, detection_threshold = 0.4, use_same_colour=True)
    print(OUTPUT_FOLDER + "/" + path.replace("/", "_"))
    Image.fromarray(image_np).save(OUTPUT_FOLDER + "/" + path.replace("/", "_"))

