import os
import torch
import numpy as np
from torch.utils import data
from tqdm import tqdm

import argparse
from pathlib import Path
from threading import Thread
from utils.log import WandbLogger

from models.facemodel import *
from attacks.functions import attack_facerecognition
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.general import increment_path

from utils.data import ImageFolderWithPaths
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid


workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def attack_5celebrity(opt):
    save_dir = increment_path(Path(opt.save_dir), exist_ok=False)  # increment run

    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=workers)
    faceverification = faceverification_retinaface_facenet()

    for image, target, path in tqdm(dataloader):
        image = image.to(device)
        att_img = attack_facerecognition(faceverification, image, None, opt)

        save_path = path.replace(opt.data, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(att_img, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='attack_lfw.py')
    parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
    parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')

    parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
    parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

    parser.add_argument('--data', type=str, default='/content/drive/MyDrive/data/5 Celebrity Faces Dataset', help='dataset')
    parser.add_argument('--pretrain-facenet', type=str, default='/content/drive/MyDrive/pretrain/face_recognition.pth', help='Path pretrain')
    
    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    opt = parser.parse_args()
    print(opt)

    attack_5celebrity(opt)