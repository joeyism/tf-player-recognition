from objects import Detection, Detections
import PIL.Image as Image
import numpy as np
import PIL.ImageDraw as ImageDraw

def to_detections(boxes, scores, classes):
    if len(boxes) != len(scores) and len(scores) != len(classes):
        raise Exception("Length of boxes, scores, and classes are different")

    total_detections = []
    n = len(boxes)
    for i in range(n):
        detection = Detection(boxes[i], scores[i], classes[i])
        total_detections.append(detection)

    return Detections(total_detections)


def draw_ellipses_around_players(image, players, fill=128):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    im_width, im_height = image_pil.size

    draw = ImageDraw.Draw(image_pil)
    for player in players:
        ymin, xmin, ymax, xmax = player.box
        (x0, x1, y0, y1) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        player_height = y1 - y0
        y0 = y1 - player_height*0.2
        draw.ellipse([x0, y0, x1, y1], fill=128)

    np.copyto(image, np.array(image_pil))
    return image
