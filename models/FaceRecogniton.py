import torch
import torch.nn as nn
from .retinaface.models.retinaface import retinaface_mnet
from facenet_pytorch import InceptionResnetV1

class FaceRecognition(nn.Module):
    def __init__(self, facedetector, facenet):
        super(FaceRecognition, self).__init__()
        self.facedetector = facedetector
        self.facenet = facenet

    def forward(self, image): 
        #TODO: what type of image 
        boxes, confidents, landmarks = self.facedetector(image)
        #TODO: what type of image
        faces = []
        for box in boxes: 
            box = box.to('cpu').tolist()
            box = [round(x) for x in box]
            #TODO: check type box
            face = torch.unsqueeze(image[box[0]:box[2], box[1]:box[3]], 0)
            face = nn.functional.interpolate(face, size=(160, 160))
            faces.append(face.sequeeze)

        faces = torch.stack(faces)

        out = self.facenet(faces)
        _, pred = torch.max(out, 1)

        return boxes, pred

def facerecognition_retinaface_facenet(pretrained=False):
    retinaface = retinaface_mnet(pretrained, phase='train')
    facenet = InceptionResnetV1(pretrained='vggface2')

    return FaceRecognition(retinaface, facenet)

