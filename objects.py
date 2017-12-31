
class Detection(object):
    box = None
    score = None
    classes = None

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

