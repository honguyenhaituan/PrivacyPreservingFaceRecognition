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
from attacks.functions import attack_faceverification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.general import increment_path

from utils.data import ImageFolderWithPaths
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid


workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def attack_lfw(opt):
    logger = WandbLogger("PrivacyPreservingFaceRecognition-lfw", None, opt) if opt.log_wandb else None
    save_dir = str(increment_path(Path(opt.save_dir), exist_ok=False))  # increment run

    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=workers)
    faceverification = faceverification_retinaface_facenet().eval().to(device)

    for image, target, path in tqdm(dataloader):
        image = image.to(device)
        att_img = attack_faceverification(faceverification, image, logger, opt)

        for _att_img, _path in zip(att_img, path): 
            save_path = _path.replace(opt.data, save_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(_att_img, save_path)

        if opt.log_wandb:
            logger.log({"sample": logger.wandb.Image(att_img[0], caption=path[0])})
            logger.end_epoch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='attack_lfw.py')
    parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
    parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')

    parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
    parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

    parser.add_argument('--data', type=str, default='/content/drive/MyDrive/data/5 Celebrity Faces Dataset', help='dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size dataloader')

    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')
    opt = parser.parse_args()
    print(opt)

    attack_lfw(opt)