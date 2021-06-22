import torch
import torch.nn as nn
from .retinaface.models.retinaface import retinaface_mnet
from .facenet import FaceNet

class FaceRecognition(nn.Module):
    def __init__(self, facedetector, facerecognition):
        super(FaceRecognition, self).__init__()
        self.facedetector = facedetector
        self.facerecognition = facerecognition

    def forward(self, image): 
        bboxes, landmarks = self.facedetector.detect_faces(image)
        if len(bboxes) == 0: 
            return None, None

        bboxes = bboxes.astype(int)
        faces = []
        for idx, boxes in enumerate(bboxes):
            for box in boxes:
                face = image[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                face = nn.functional.interpolate(face, size=(160, 160))
                faces.append(face.squeeze())

        faces = torch.stack(faces)

        out = self.facerecognition(faces)
        _, pred = torch.max(out, 1)

        return (torch.from_numpy(bboxes).to(pred.device), torch.from_numpy(landmarks).to(pred.device)), pred

def facerecognition_retinaface_facenet(pretrained=None, num_classes=None):
    retinaface = retinaface_mnet(pretrained=True)
    facenet = FaceNet(pretrained=pretrained, num_classes=num_classes)

    return FaceRecognition(retinaface, facenet)