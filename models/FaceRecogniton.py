import torch
import torch.nn as nn
from .retinaface.models.retinaface import retinaface_mnet
from facenet_pytorch import InceptionResnetV1

class FaceRecognition(nn.Module):
    def __init__(self, facedetector, facerecognition):
        super(FaceRecognition, self).__init__()
        self.facedetector = facedetector
        self.facerecognition = facerecognition

    def forward(self, image): 
        boxes, confidents, landmarks = self.facedetector.detect_faces(image)
        faces = []
        for box in boxes: 
            box = [round(x) for x in box]
            #TODO: check type box
            face = image[:, :, box[0]:box[2], box[1]:box[3]]
            face = nn.functional.interpolate(face, size=(160, 160))
            faces.append(face.squeeze())

        faces = torch.stack(faces)

        out = self.facerecognition(faces)
        _, pred = torch.max(out, 1)

        return boxes, pred

def facerecognition_retinaface_facenet(pretrained=False):
    retinaface = retinaface_mnet(pretrained, phase='train')
    facenet = InceptionResnetV1(pretrained='vggface2')

    return FaceRecognition(retinaface, facenet)
