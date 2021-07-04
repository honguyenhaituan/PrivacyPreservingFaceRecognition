from models.extractor import Extractor
import torch
import numpy as np
import torch.nn as nn
from .detector import Detector, RetinaFaceDetector
from .extractor import ArcFaceExtractor, FaceNet, FaceNetExtractor, CosFaceExtractor

class FaceVerification(nn.Module):
    def __init__(self, detector: Detector, extractor: Extractor):
        super(FaceVerification, self).__init__()
        self.detector = detector
        self.extactor = extractor

    def forward(self, image): 
        bboxes, landmarks = self.detector.detect(image)
        out = self.embedding(image, bboxes)

        return (bboxes, landmarks), out

    def embedding(self, image, bboxes):
        faces = []
        for idx, boxes in enumerate(bboxes):
            for box in boxes:
                face = image[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                face = nn.functional.interpolate(face, size=self.extactor.input_size)
                faces.append(face.squeeze())
            if len(boxes) == 0:
                face = image[idx:idx + 1]
                face = nn.functional.interpolate(face, size=self.extactor.input_size)
                faces.append(face.squeeze())

        faces = torch.stack(faces)
        return self.extactor(faces)

def faceverification_retinaface_facenet(pretrained='vggface2'):
    retinaface = RetinaFaceDetector()
    facenet = FaceNetExtractor(pretrained=pretrained)

    return FaceVerification(retinaface, facenet)

def faceverification_retinaface_arcface(backbone='r18'):
    detetor = RetinaFaceDetector()
    extractor = ArcFaceExtractor(backbone)
    
    return FaceVerification(detetor, extractor)

def faceverification_retinaface_cosface(backbone='r18'):
    detetor = RetinaFaceDetector()
    extractor = CosFaceExtractor(backbone)
    
    return FaceVerification(detetor, extractor)

class FaceRecognition(FaceVerification):
    def forward(self, image):
        face, logit = super().forward(image)
        _, pred = torch.max(logit, 1)
        return face, pred

def facerecognition_retinaface_facenet(pretrained='vggface2', num_classes=None):
    retinaface = RetinaFaceDetector()
    facenet = FaceNet(pretrained=pretrained, classify=True, num_classes=num_classes)

    return FaceRecognition(retinaface, facenet)