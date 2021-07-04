from torch.nn import nn 
from facenet_pytorch import MTCNN, training
from .retinaface.models.retinaface import retinaface_mnet

from typing import Tuple, List


class Detector(nn.Moudule):
    def detect(self, inputs, isOut=False) -> Tuple[List, List]:
        pass

class RetinaFaceDetector(Detector):
    def __init__(self):
        super(Detector).__init__()
        self.retinaface = retinaface_mnet(pretrained=True)

    def forward(self, image):
        self.shape = image.shape
        return self.retinaface(image)
        
    def detect(self, inputs, isOut=False):
        if isOut: 
            return self.retinaface.select_boxes(inputs, self.shape)
    
        return self.retinaface.detect(inputs)

class MTCNNDetector(Detector):
    def __init__(self):
        super(Detector).__init__()
        self.mtcnn = MTCNN()

    def _transform(inputs):
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
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, inputs, method=self.mtcnn.selection_method
            )

        return batch_boxes, batch_points