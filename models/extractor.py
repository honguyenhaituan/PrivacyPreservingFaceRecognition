import torch
import torch.nn as nn
from .backbones import get_model
from facenet_pytorch import InceptionResnetV1

def fixed_image_standardization(image_tensor):
    processed_tensor = image_tensor * 2 - 1
    return processed_tensor

class Extractor(nn.Module):
    def __init__(self, model, input_size, transform):
        super(Extractor, self).__init__()
        self.model = model
        self.input_size = input_size
        self.transform = transform
        self.margin = 20

    def forward(self, image):
        image = self.transform(image)
        return self.model(image)

def FaceNetExtractor(pretrained='vggface2'): 
    facenet = InceptionResnetV1(pretrained)
    return Extractor(facenet, (160, 160), fixed_image_standardization)

def ArcFaceExtractor(backbone='r18'):
    if backbone == 'r18': 
        path = "https://github.com/honguyenhaituan/PrivacyPreservingFaceRecognition/releases/download/v1/ms1mv3_arcface_r18_fp16.pth"
    else:
        raise ValueError("Arcface hasn't support backbone {} pretrained yet.".format(backbone))

    arcface = get_model(backbone, fp16=False)
    state_dict = torch.hub.load_state_dict_from_url(path)
    arcface.load_state_dict(state_dict)
    return Extractor(arcface, (112, 112), fixed_image_standardization)

def CosFaceExtractor(backbone='r18'):
    if backbone == 'r18':
        path = "https://github.com/honguyenhaituan/PrivacyPreservingFaceRecognition/releases/download/v1/glint360k_cosface_r18_fp16_0.1.pth"
    else:
        raise ValueError("Cosface hasn't support backbone {} pretrained yet.".format(backbone))
        
    cosface = get_model(backbone, fp16=False)
    state_dict = torch.hub.load_state_dict_from_url(path)
    cosface.load_state_dict(state_dict)
    return Extractor(cosface, (112, 112), fixed_image_standardization)

def get_extractor(name):
    if name == 'facenet':
        return FaceNetExtractor()
    elif name == 'arcface':
        return ArcFaceExtractor()
    elif name == 'cosface':
        return CosFaceExtractor()
    else:
        raise ValueError("Name extractor dont support")

class FaceNet(InceptionResnetV1):
    def forward(self, x):
        x = fixed_image_standardization(x)
        return super().forward(x)
