import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, training
from .retinaface.models.retinaface import retinaface_mnet

from typing import Tuple, List


class Detector(nn.Module):
    def detect(self, inputs, isOut=False) -> Tuple[List, List]:
        pass

class RetinaFaceDetector(Detector):
    def __init__(self):
        super(Detector, self).__init__()
        self.retinaface = retinaface_mnet(pretrained=True)

    def forward(self, image):
        self.shape = image.shape
        return self.retinaface(image)
        
    def detect(self, inputs, isOut=False):
        if isOut: 
            return self.retinaface.select_boxes(inputs, self.shape)
    
        return self.retinaface.detect(inputs)

class MTCNNDetector(Detector):
    def __init__(self, device='cpu'):
        super(Detector, self).__init__()
        self.mtcnn = MTCNN(device=device)

    def _transform(self, inputs):
        inputs = inputs * 255
        inputs = inputs.permute(0, 2, 3, 1)
        return inputs

    def detect(self, inputs, isOut=False):
        if isOut: 
            raise ValueError("MTCNN dont take output detect")

        inputs = self._transform(inputs)

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(inputs, landmarks=True)
        # Select faces
        if not self.mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, inputs, method=self.mtcnn.selection_method
            )

        boxes, lands = [], []
        for box, land in zip(batch_boxes, batch_points):
            boxes.append(torch.from_numpy(box.astype(int)))
            lands.append(torch.from_numpy(land))

        return boxes, lands

def get_detector(name):
    if name == 'retinaface':
        return RetinaFaceDetector()
    elif name == 'mtcnn':
        return MTCNNDetector(device='cuda')
    else:
        raise ValueError("Name detector dont support")