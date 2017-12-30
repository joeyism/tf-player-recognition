import tensorflow as tf
import os
from utils import label_map_util
from PIL import Image
import numpy as np

def load_image_into_numpy_array(image):
    (im_height, im_width, channels) = np.array(image).shape
    image_np = np.array(image.getdata()).reshape((im_height, im_width, channels))
    if channels > 3:
        image_np = image_np[:, :, :3]
    return image_np.astype(np.uint8)


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

TEST_IMAGES_FOLDER = "sports_images"
OUTPUT_FOLDER = "output"
TEST_IMAGES_PATHS = [ os.path.join(TEST_IMAGES_FOLDER, image_path) for image_path in os.listdir(TEST_IMAGES_FOLDER)]

with detection_graph.as_default():
    sess = tf.Session(graph = detection_graph)
    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
    detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
    detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
    detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    for i, image_path in enumerate(TEST_IMAGES_PATHS):
        print(i)
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = sess.run([
            detection_boxes,
            detection_scores,
            detection_classes,
            num_detections
            ],
            feed_dict = { image_tensor: image_np_expanded }
            )

       Image.fromarray(image_np).save(OUTPUT_FOLDER + "/test"+str(i) + ".jpg")




