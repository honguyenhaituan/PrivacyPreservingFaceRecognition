import os
import torch
import numpy as np
from tqdm import tqdm

import argparse
from pathlib import Path
from threading import Thread
from utils.log import WandbLogger

from models.facemodel import faceverification_retinaface_facenet

from utils.general import increment_path
from utils.data import ImageFolderWithPaths
from utils.lfw_file import read_pairs, get_paths
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.metrics import evaluate_lfw
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def attack_lfw(opt):
    # logger = WandbLogger("PrivacyPreservingFaceRecognition-lfw", None, opt) if opt.log_wandb else None
    save_dir = str(increment_path(Path(opt.save_dir), exist_ok=False))  # increment run

    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=workers)
    faceverification = faceverification_retinaface_facenet().eval().to(device)

    classes, embeddeds, paths = [], [], []
    for image, target, path in tqdm(dataloader):
        image = image.to(device)
        _, embedded = faceverification(image)
        embedded = embedded.to('cpu').numpy()

        classes.extend(target.numpy())
        embeddeds.extend(embedded)
        paths.extend(np.asarray(path))

    embeddeds_dict = dict(zip(paths,embeddeds))
    pairs = read_pairs(opt.pairs_path)
    path_list, issame_list = get_paths(opt.data, pairs)
    embeddeds = np.array([embeddeds_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate_lfw(embeddeds, issame_list,
                                                                distance_metric=opt.distance_metric,
                                                                subtract_mean=opt.subtract_mean)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='validate_lfw.py')
    parser.add_argument('--data', type=str, default='./data/lfw', help='dataset')
    parser.add_argument('--pairs-path', type=str, default='./data/pairs.txt', help='pair file to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size dataloader')

    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')

    parser.add_argument('--distance_metric', type=int, help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--subtract_mean', help='Subtract feature mean before calculating distance.', action='store_true')
    opt = parser.parse_args()
    print(opt)

    attack_lfw(opt)