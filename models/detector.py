import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN
from .retinaface.models.retinaface import retinaface_mnet

from typing import Tuple, List


class Detector(nn.Module):
    def detect(self, inputs, isOut=False) -> Tuple[List, List]:
        pass

class RetinaFaceDetector(Detector):
    def __init__(self):
        super(Detector, self).__init__()
        self.retinaface = retinaface_mnet(pretrained=True)
        self.cfg = self.retinaface.cfg

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

    def select_boxes(self, all_boxes, all_probs, all_points, imgs, center_weight=2.0):

        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):
            
            if boxes is None:
                selected_boxes.append(np.array([]))
                selected_probs.append(np.array([]))
                selected_points.append(np.array([]))
                continue
            
            box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            img_center = (img.shape[1] / 2, img.shape[0]/2)
            box_centers = np.array(list(zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)))
            offsets = box_centers - img_center
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
            box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        selected_boxes = np.array(selected_boxes)
        selected_probs = np.array(selected_probs)
        selected_points = np.array(selected_points)

        return selected_boxes, selected_probs, selected_points

    def detect(self, inputs, isOut=False):
        if isOut: 
            raise ValueError("MTCNN dont take output detect")

        inputs = self._transform(inputs)

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(inputs, landmarks=True)
        # Select faces
        if not self.mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, inputs
            )

        boxes, lands = [], []
        for box, land in zip(batch_boxes, batch_points):
            if len(box) != 0:
                boxes.append(torch.from_numpy(box.astype(int)))
            else:
                boxes.append(torch.from_numpy(box))

            lands.append(torch.from_numpy(land))

        return boxes, lands

def get_detector(name):
    if name == 'retinaface':
        return RetinaFaceDetector()
    elif name == 'mtcnn':
        return MTCNNDetector(device='cuda')
    else:
        raise ValueError("Name detector dont support")