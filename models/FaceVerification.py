import torch
import torch.nn as nn
from .retinaface.models.retinaface import retinaface_mnet
from facenet_pytorch import InceptionResnetV1

class FaceVerification(nn.Module):
    def __init__(self, facedetector, backbone):
        super(FaceVerification, self).__init__()
        self.facedetector = facedetector
        self.backbone = backbone

    def embeded(self, image): 
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

        out = self.backbone(faces)

        return (torch.from_numpy(bboxes).to(out.device), torch.from_numpy(landmarks).to(out.device)), out

def faceverification_retinaface_facenet(pretrained=None):
    retinaface = retinaface_mnet(pretrained=True)
    facenet = InceptionResnetV1(pretrained=pretrained)

    return FaceVerification(retinaface, facenet)