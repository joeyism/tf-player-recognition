from objects import Detection, Detections
from sklearn.cluster import KMeans
import PIL.Image as Image
import numpy as np
import PIL.ImageDraw as ImageDraw

boundaries = [
            ([17, 15, 100], [50, 56, 200]), #blue
            ([80, 0, 0], [255, 133, 133]), #red
            ([86, 31, 4], [220, 88, 50]), #brown/orange
            ([25, 146, 190], [62, 174, 250]), #blue
            ([103, 86, 65], [145, 133, 128]), #grey
            ([190,190,190], [255, 255, 255]) # white
        ]

def in_range(tup, boundary):
    lower = boundary[0]
    upper = boundary[1]
    res = []
    for i, val in enumerate(tup):
        res.append(True if val >= lower[i] and val <= upper[i] else False)
    return all(res)

def get_boundary_num(tup):
    for i, boundary in enumerate(boundaries):
        if in_range(tup, boundary):
            return i
    return -1

def to_detections(image, boxes, scores, classes):
    if len(boxes) != len(scores) and len(scores) != len(classes):
        raise Exception("Length of boxes, scores, and classes are different")

    total_detections = []
    n = len(boxes)
    im_width, im_height = image.size

    for i in range(n):
        detection = Detection(boxes[i], scores[i], classes[i])
        ymin, xmin, ymax, xmax = boxes[i]
        detection.normalized_box = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        detection.box_image = image.crop((detection.normalized_box[0], detection.normalized_box[2], detection.normalized_box[1], detection.normalized_box[3]))
        detection.center = ( im_width*(xmin + xmax)/2 , im_height*(ymin + ymax)/2  )

        total_detections.append(detection)

    return Detections(total_detections)


def load_image_into_numpy_array(image):
    (im_height, im_width, channels) = np.array(image).shape
    image_np = np.array(image.getdata()).reshape((im_height, im_width, channels))
    if channels > 3:
        image_np = image_np[:, :, :3]
    return image_np.astype(np.uint8)


def draw_ellipses_around_players(image, players):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    for player in players:
        (x0, x1, y0, y1) = player.normalized_box
        player_height = y1 - y0
        y0 = y1 - player_height*0.2
        #print(player.colour)
        draw.ellipse([x0, y0, x1, y1], fill=player.colour)

    np.copyto(image, np.array(image_pil))
    return image

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def get_colours_from_image(image):
    height, width, dim = image.shape
    img_vec = np.reshape(image, [height * width, dim] )
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(img_vec)
    score = centroid_histogram(kmeans)
    colours = []
    scores = []
    for i, center in enumerate(kmeans.cluster_centers_):
        colour = (int(center[0]), int(center[1]), int(center[2]))
        if colour == (0,0,0):
            continue
        colours.append(colour)
        scores.append(score[i])
    return colours[np.argsort(scores)[-2]]

def set_colours_on_detections(detections):
    for detection in detections:
        detection.colour = get_colours_from_image(load_image_into_numpy_array(detection.box_image))
        detection.boundary_index = get_boundary_num(detection.colour)

