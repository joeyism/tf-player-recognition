import math
import numpy as np

class Detection(object):
    box = None
    score = None
    classes = None
    normalized_box = None
    box_image = None
    colour = None
    colour_lab_index = -1
    boundary_index = None
    center = None

    def __init__(self, box, score, classes):
        self.box = box
        self.score = score
        self.classes = classes

    def __repr__(self):
        return "box: {box}\nscore: {score}\nclasses: {classes}\nboundary_index: {boundary_index}\ncolour: {colour}\n\n".format(
                box = self.box,
                score = self.score,
                classes = self.classes,
                boundary_index = self.boundary_index,
                colour = self.colour
                )

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

    def filter_classify(self, classify):
        result = []
        for detection in self:
            if detection.classify == classify:
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

class Frames(list):
    BATCH_SIZE = 16

    def get_batch(self, i):
        length_of_frames = len(self)
        size = self[0].shape
        max_batch_no = int(length_of_frames/self.BATCH_SIZE) + 1
        result = self[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
        while len(result) < self.BATCH_SIZE:
            result.append(np.zeros(size))
        return result


