from objects import Detection, Detections
from sklearn.cluster import KMeans
import PIL.Image as Image
import numpy as np
import PIL.ImageDraw as ImageDraw

# boundaries in the form of (lower_limit, upper_limit, appropriate_color)
boundaries = [
            ([60, 60, 60], [80, 80, 80], (100, 100, 100)), # greys
            ([70, 70, 70], [95, 95, 95], (100, 100, 100)),
            ([80, 80, 80], [100, 100, 100], (100, 100, 100)),
            ([103, 86, 65], [145, 133, 128], (100, 100, 100)),
            ([190,190,190], [255, 255, 255], (255, 255, 255)), # white
            ([50, 0, 0], [255, 200, 110], (255, 0, 0)), #red
            ([0, 0, 60], [105, 240, 255], (0, 0, 255)), #blue
            ([10, 30, 30], [50, 100, 100], (50, 100, 100)), #teal
            ([0, 0, 0], [1, 1, 1], (0, 0, 0))
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

        total_detections.append(detection)

    return Detections(total_detections)


def load_image_into_numpy_array(image):
    (im_height, im_width, channels) = np.array(image).shape
    try:
        image_np = np.array(image.getdata()).reshape((im_height, im_width, channels))
    except:
        return image

    if channels > 3:
        image_np = image_np[:, :, :3]
    return image_np.astype(np.uint8)

def diffs(x0, x1, y0, y1):
    return (x1 - x0, y1 - y0)


def draw_ellipses_around_players(image, players):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    for player in players:
        (x0, x1, y0, y1) = player.normalized_box
        if x1 - x0 > image.shape[1]/3 or y1 - y0 > image.shape[0]/3:
            continue
        player_height = y1 - y0
        y0 = y1 - player_height*0.2
        draw.ellipse([x0, y0, x1, y1], fill=player.colour)
        draw.text(player.center, str(player.boundary_index), fill=(255, 255, 255))

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

def set_colours_on_detections(detections, use_same_colour = True):
    for detection in detections:
        detection.colour = get_colours_from_image(load_image_into_numpy_array(detection.box_image))
        detection.boundary_index = int(get_boundary_num(detection.colour))
        if use_same_colour and detection.boundary_index >= 0:
            detection.colour = boundaries[detection.boundary_index][2]

def draw_line_from_distances(image, players_group, distances):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    draw = ImageDraw.Draw(image_pil)
    for distance in distances:
        n1 = int(distance[0])
        n2 = int(distance[1])
        draw.line(
                [ players_group[n1].center, players_group[n2].center ], 
                fill=players_group[n1].colour,
                width = 3
                )
    np.copyto(image, np.array(image_pil))
    return image

def draw_lines_between_players(image, players):
    no_boundaries = len(boundaries)
    for i in range(no_boundaries):
        players_group = players.filter_boundary_index(i)
        n = len(players_group)
        distances = players_group.distances()
        if len(distances) == 0:
            continue
        distances = distances[:n-1]
        image = draw_line_from_distances(image, players_group, distances)

