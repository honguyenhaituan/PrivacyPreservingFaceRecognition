import torch
import numpy as np
import torch.nn as nn
from .retinaface.models.retinaface import retinaface_mnet
from .facenet import FaceNet

class FaceVerification(nn.Module):
    def __init__(self, facedetector, facerecognition):
        super(FaceVerification, self).__init__()
        self.facedetector = facedetector
        self.facerecognition = facerecognition

    def forward(self, image): 
        bboxes, landmarks = self.facedetector.detect_faces(image)

        faces = []
        for idx, boxes in enumerate(bboxes):
            for box in boxes:
                face = image[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                face = nn.functional.interpolate(face, size=(160, 160))
                faces.append(face.squeeze())
            if len(boxes) == 0:
                face = image[idx:idx + 1]
                face = nn.functional.interpolate(face, size=(160, 160))
                faces.append(face.squeeze())

        faces = torch.stack(faces)
        out = self.facerecognition(faces)

        return (bboxes, landmarks), out

class FaceRecognition(FaceVerification):
    def forward(self, image):
        face, logit = super().forward(image)
        _, pred = torch.max(logit, 1)
        return face, pred

def facerecognition_retinaface_facenet(pretrained='vggface2', num_classes=None):
    retinaface = retinaface_mnet(pretrained=True)
    facenet = FaceNet(pretrained=pretrained, classify=True, num_classes=num_classes)

    return FaceRecognition(retinaface, facenet)

def faceverification_retinaface_facenet(pretrained='vggface2'):
    retinaface = retinaface_mnet(pretrained=True)
    facenet = FaceNet(pretrained=pretrained)

    return FaceVerification(retinaface, facenet)