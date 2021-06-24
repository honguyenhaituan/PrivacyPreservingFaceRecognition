import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from models.facemodel import *

from utils.data import ImageFolderWithPaths
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils.general import increment_path


workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def select_CASIAWebFace(opt): 
    save_dir = str(increment_path(Path(opt.save_dir), exist_ok=False))

    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, num_workers=workers)
    facerecognition = facerecognition_retinaface_facenet(pretrained='casia-webface').eval().to(device)

    cnt = {}
    for image, target, path in tqdm(dataloader):
        label = target.item()
        if label not in cnt: cnt[label] = 0
        if cnt[label] >= opt.count:
            continue

        image = image.to(device)
        _, pred = facerecognition(image)

        if pred.item() == label:
            save_path = path[0].replace(opt.data, save_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(image, save_path)
            cnt[label] += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='select_CASIAWebFace.py')
    parser.add_argument('--data', type=str, default='./data/CASIA-WebFace', help='dataset')
    parser.add_argument('--save-dir', type=str, default='./data/CASIA-WebFace-mini', help='Dir save WebFace')
    parser.add_argument('--count', type=int, default=1, help='Amount image per class selectsa')
    opt = parser.parse_args()
    print(opt)

    select_CASIAWebFace(opt)