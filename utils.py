from objects import Detection, Detections
from sklearn.cluster import KMeans
import PIL.Image as Image
import numpy as np
import PIL.ImageDraw as ImageDraw
from timeme import time_here
from skimage.color import rgb2lab, deltaE_cie76

# boundaries in the form of (lower_limit, upper_limit, appropriate_color)
boundaries = [
            ([60, 60, 60], [80, 80, 80], (100, 100, 100)), # greys
            ([70, 70, 70], [95, 95, 95], (100, 100, 100)),
            ([80, 80, 80], [100, 100, 100], (100, 100, 100)),
            ([103, 86, 65], [145, 133, 128], (100, 100, 100)),
            ([190,190,190], [255, 255, 255], (255, 255, 255)), # white
            ([50, 0, 0], [255, 200, 110], (255, 0, 0)), #red
            ([0, 0, 60], [145, 240, 255], (0, 0, 255)), #blue
            ([10, 30, 30], [50, 100, 100], (50, 100, 100)), #teal
            ([0, 0, 0], [1, 1, 1], (0, 0, 0))
        ]

colours_lab = [
    ( rgb2lab([[[0,0,45]]]), [0,0,45] ),
    ( rgb2lab([[[0,0,128]]]), [0,0,128] ),
    ( rgb2lab([[[0,0,255]]]), [0,0,255] ),
    ( rgb2lab([[[45,0,0]]]), [45,0,0] ),
    ( rgb2lab([[[128,0,0]]]), [128,0,0] ),
    ( rgb2lab([[[255,0,0]]]), [255,0,0] ),
    ( rgb2lab([[[0,255,0]]]), [0,255,0] ),
    ( rgb2lab([[[135, 206, 235]]]), [0, 0, 255] ),
    ( rgb2lab([[[255,255,255]]]), [255,255,255] ),
    ( rgb2lab([[[0,0,0]]]), [255,255,255] )
]

def normalize_colours(crop_np, threshold=10):
    for colour_lab, original_colour in colours_lab:
        crop_lab = rgb2lab(crop_np)
        distances = deltaE_cie76(colour_lab, crop_lab)
        crop_np[distances < threshold] = original_colour

    return crop_np

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
        total_detections.append(detection)

    return Detections(total_detections)


def load_image_into_numpy_array(image):
    (im_height, im_width, channels) = np.array(image).shape
    try:
        image_np = np.array(image)
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
    kmeans = KMeans(n_clusters=2, n_init=2, max_iter=10, precompute_distances=True, algorithm="elkan", random_state = 0).fit(img_vec)
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

def set_colour_on_detection(detection, use_same_colour = True):
    detection.colour = get_colours_from_image(load_image_into_numpy_array(detection.box_image))
    detection.boundary_index = int(get_boundary_num(detection.colour))
    if use_same_colour and detection.boundary_index >= 0:
        detection.colour = boundaries[detection.boundary_index][2]
    return detection

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

def add_detected_image(image, player, colour_threshold = 20):
    im_width, im_height = image.size
    ymin, xmin, ymax, xmax = player.box
    player_height = ymax - ymin

    player.normalized_box = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height
        )
    player.box_image = image.crop((
        int(player.normalized_box[0]),
        int(player.normalized_box[2]),
        int(player.normalized_box[1]),
        int(player.normalized_box[3] - player_height*im_height/2)
        ))
    player.center = ( im_width*(xmin + xmax)/2 , im_height*(ymax - player_height*0.1)  )
    image_np = normalize_colours(load_image_into_numpy_array(player.box_image), threshold = colour_threshold)
    player.box_image = Image.fromarray(image_np)
    return player

def add_to_detection(image, player, colour_threshold = 20):
    player = add_detected_image(image, player, colour_threshold = 20)
    player = set_colour_on_detection(player)
    return player
