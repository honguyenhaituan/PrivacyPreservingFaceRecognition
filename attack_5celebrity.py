import os
import torch
import numpy as np
from tqdm import tqdm

import argparse
from pathlib import Path
from threading import Thread
from utils.log import WandbLogger

from torchvision import datasets, transforms
from models.FaceRecogniton import *
from attacks.attacks import attack_facerecognition
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def attack_5celebrity(opt, logger):
    workers = 0 if os.name == 'nt' else 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data, x),
                                          transform=transforms.ToTensor())
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, 
                                                num_workers=workers)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    retinaface = retinaface_mnet(pretrained=True).eval()
    facenet = InceptionResnetV1(classify=True, num_classes=len(class_names)).eval()
    facenet.load_state_dict(torch.load(opt.pretrain_facenet))
    facerecognition = FaceRecognition(retinaface, facenet).to(device)

    pred, label = [], []
    for image, target in dataloaders["val"]:
        image = image.to(device)
        att_img = attack_facerecognition(facerecognition, image, 'MI-FGSM', 20, 0.9, None)
        bbox, name = facerecognition(att_img)

        for _pred, _label in zip(name, target):
            pred.append(_pred.item())
            label.append(_label.item())

    logger.log({"metrics/accuracy": accuracy_score(target, pred)})
    logger.log({"metrics/f1": f1_score(target, pred)})
    logger.log({"metrics/precision": precision_score(target, pred)})
    logger.log({"metrics/recall": recall_score(target, pred)})
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='attack_5celebrity.py')
    parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
    parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')
    parser.add_argument('--type_blur', action='store_true', help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')

    parser.add_argument('--data', type=str, default='/content/drive/MyDrive/data/5 Celebrity Faces Dataset', help='dataset')
    parser.add_argument('--pretrain-facenet', type=str, default='/content/drive/MyDrive/pretrain/face_recognition.pth', help='Path pretrain')
    parser.add_argument('--save-attack-image', action='store_true', help='Save image file after attack')
    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    
    opt = parser.parse_args()
    print(opt)

    wandb_logger = WandbLogger("PrivacyPreservingFaceRecognition/5celebrity", None, opt)
    attack_5celebrity(opt, wandb_logger)