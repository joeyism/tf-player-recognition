from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import os
import label_map_util
import utils
import sys
from timeme import time_here

time_here("1")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph = tf.GraphDef()
    with tf.gfile.GFile("ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb", "rb") as fid:
        serialized_graph = fid.read()
        od_graph.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph, name="")

time_here("2")

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

OUTPUT_FOLDER = "output"

time_here("3")

detection_graph.as_default()
sess = tf.Session(graph = detection_graph)
image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections = detection_graph.get_tensor_by_name("num_detections:0")

time_here("4")

def detect_image(image, threshold = 0.4, use_same_colour = True):
    im_width, im_height = image.size
    old_image = image.copy()
    old_image_np = utils.load_image_into_numpy_array(old_image)

    # detection
    time_here("7")
    image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.5)).enhance(1.5)
    time_here("8")
    image_np = utils.load_image_into_numpy_array(image)
    time_here("9")
    image_np_expanded = np.expand_dims(image_np, axis=0)
    time_here("10")
    (boxes, scores, classes, num) = sess.run([
        detection_boxes,
        detection_scores,
        detection_classes,
        num_detections
        ],
        feed_dict = { image_tensor: image_np_expanded }
        )
    time_here("11")

    detections = utils.to_detections(image, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))
    time_here("12")
    detections = detections.filter_classes(1).filter_score(gt=threshold)

    for i, detection in enumerate(detections):
        ymin, xmin, ymax, xmax = detection.box
        player_height = ymax - ymin

        detection.normalized_box = (
                xmin * im_width,
                xmax * im_width,
                ymin * im_height,
                ymax * im_height
            )
        detection.box_image = image.crop((
            int(detection.normalized_box[0]),
            int(detection.normalized_box[2]),
            int(detection.normalized_box[1]),
            int(detection.normalized_box[3] - player_height*im_height/2)
            ))
        detection.center = ( im_width*(xmin + xmax)/2 , im_height*(ymax - player_height*0.1)  )
        detection.box_image = Image.fromarray(utils.normalize_colours(np.array(detection.box_image)))
        detection.box_image.save("test{}.jpg".format(i))

    time_here("13")
    utils.set_colours_on_detections(detections, use_same_colour = use_same_colour)
    time_here("14")

    # addons
    utils.draw_ellipses_around_players(old_image_np, detections)
    time_here("15")
    utils.draw_lines_between_players(old_image_np, detections)
    time_here("16")

    return old_image_np

def test():
    TEST_IMAGES_FOLDER = "sports_images"
    TEST_IMAGES_PATHS = [ os.path.join(TEST_IMAGES_FOLDER, image_path) for image_path in os.listdir(TEST_IMAGES_FOLDER)]
    for i, image_path in enumerate(TEST_IMAGES_PATHS):
        print(i, image_path)
        image = Image.open(image_path)
        image_np = detect_image(image, use_same_colour = False)
        Image.fromarray(image_np).save(OUTPUT_FOLDER + "/" + image_path.replace("/", "_"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
        sys.exit()
    path = sys.argv[1]
    time_here("5")
    image = Image.open(path)
    time_here("6")
    image_np = detect_image(image, threshold = 0.4, use_same_colour=True)
    time_here("17")
    Image.fromarray(image_np).save(OUTPUT_FOLDER + "/" + path.replace("/", "_"))

