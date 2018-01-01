import math
import numpy as np

class Detection(object):
    box = None
    score = None
    classes = None
    normalized_box = None
    box_image = None
    colour = None
    boundary_index = None
    center = None

    def __init__(self, box, score, classes):
        self.box = box
        self.score = score
        self.classes = classes

    def __repr__(self):
        return "box: {box}\nscore: {score}\nclasses: {classes}".format(
                box = self.box,
                score = self.score,
                classes = self.classes)

class Detections(list):

    def filter_score(self, gt=None, lt=None):
        result = []
        for detection in self:
            if (gt and detection.score > gt) or (lt and detection.score < lt):
                result.append(detection)
        return Detections(result)

    def filter_classes(self, classes):
        result = []
        for detection in self:
            if detection.classes == classes:
                result.append(detection)
        return Detections(result)

    def filter_boundary_index(self, index):
        result = []
        for detection in self:
            if detection.boundary_index == index:
                result.append(detection)
        return Detections(result)

    def distances(self):
        n = len(self)
        result = []

        for i in range(n):
            p1 = self[i].center
            for j in range(i+1, n):
                p2 = self[j].center
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                result.append([i, j, distance])
        return np.array(result)
                


