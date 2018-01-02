from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import os
import label_map_util
import utils
import sys
import multiprocessing as mp
from objects import Detections
from functools import partial
from timeme import time_here

no_cpu = mp.cpu_count()
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph = tf.GraphDef()
    with tf.gfile.GFile("ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb", "rb") as fid:
        serialized_graph = fid.read()
        od_graph.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph, name="")

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

OUTPUT_FOLDER = "output"


detection_graph.as_default()
sess = tf.Session(graph = detection_graph)
image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections = detection_graph.get_tensor_by_name("num_detections:0")


def detect_image(image, threshold = 0.4, use_same_colour = True):
    old_image = image.copy()
    old_image_np = utils.load_image_into_numpy_array(old_image)

    # detection
    image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.5)).enhance(1.5)
    image_np = utils.load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run([
        detection_boxes,
        detection_scores,
        detection_classes,
        num_detections
        ],
        feed_dict = { image_tensor: image_np_expanded }
        )

    detections = utils.to_detections(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))
    detections = detections.filter_classes(1).filter_score(gt=threshold)

    pool = mp.Pool(no_cpu)
    func = partial(utils.add_to_detection, image)
    detections = pool.map(func, detections)
    detections = Detections(detections)

    # addons

    utils.draw_ellipses_around_players(old_image_np, detections)
    utils.draw_lines_between_players(old_image_np, detections)

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
    image_np = detect_image(image, threshold = 0.4, use_same_colour=True)
    Image.fromarray(image_np).save(OUTPUT_FOLDER + "/" + path.replace("/", "_"))

