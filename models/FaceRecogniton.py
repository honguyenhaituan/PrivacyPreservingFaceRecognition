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
        boxes, landmarks = self.facedetector.detect_faces(image)
        boxes = boxes.astype(int)
        faces = []
        for box in boxes:
            face = image[:, :, box[1]:box[3], box[0]:box[2]]
            face = nn.functional.interpolate(face, size=(160, 160))
            faces.append(face.squeeze())

        faces = torch.stack(faces)

        out = self.facerecognition(faces)
        _, pred = torch.max(out, 1)

        return torch.from_numpy(boxes).unsqueeze(0), pred

def facerecognition_retinaface_facenet(pretrained=False):
    retinaface = retinaface_mnet(pretrained, phase='train')
    facenet = InceptionResnetV1(pretrained='vggface2')

    return FaceRecognition(retinaface, facenet)