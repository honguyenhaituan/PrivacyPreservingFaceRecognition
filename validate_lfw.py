from models.retinaface import utils
import os
import torch
import numpy as np
import sklearn
from tqdm import tqdm

import argparse
from pathlib import Path

from utils.general import increment_path
from utils.data import ImageFolderWithPaths
from utils.lfw_file import read_pairs, get_paths
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.metrics import evaluate_lfw
from utils.log import WandbLogger
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from models.extractor import get_extractor
from models.detector import get_detector
from models.facemodel import FaceVerification

workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def attack_lfw(opt):
    logger = WandbLogger("PrivacyPreservingFaceRecognition-lfw-evaluate", None, opt) if opt.log_wandb else None
    # save_dir = str(increment_path(Path(opt.save_dir), exist_ok=False))  # increment run

    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=workers)

    detector = get_detector(opt.detector)
    extractor = get_extractor(opt.extractor)
    faceverification = FaceVerification(detector, extractor).eval().to(device)

    classes, embeddeds, paths = [], [], []
    for image, target, path in tqdm(dataloader):
        image = image.to(device)
        _, embedded = faceverification(image)
        embedded = embedded.to('cpu').numpy()

        if opt.flip:
            image_flip = torch.flip(image, [-1])
            _, embedded_flip = faceverification(image_flip)
            embedded_flip = embedded_flip.to('cpu').numpy()
            embedded += embedded_flip

        classes.extend(target.numpy())
        embeddeds.extend(embedded)
        paths.extend(np.asarray(path))

    embeddeds_dict = dict(zip(paths,embeddeds))
    pairs = read_pairs(opt.pairs_path)
    path_list, issame_list = get_paths(opt.data, pairs)
    embeddeds = np.array([embeddeds_dict[path] for path in path_list])
    embeddeds = sklearn.preprocessing.normalize(embeddeds)

    tpr, fpr, accuracy, val, val_std, far, fp, fn, threshold, predict = evaluate_lfw(embeddeds, issame_list,
                                                                distance_metric=opt.distance_metric,
                                                                subtract_mean=opt.subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    print('Thresh hold', threshold)
    print('False positive', np.sum(fp)/len(fp))
    print('False negative', np.sum(fn)/len(fn))

    if logger:
        logger.log({'Accuracy mean': np.mean(accuracy)})
        logger.log({'Accuracy std': np.std(accuracy)})
        logger.log({'Validation rate': val})
        logger.log({'Validation rate std': val_std})
        logger.log({'FAR': far})
        logger.log({'AUC': auc})
        logger.log({'eer': eer})
        logger.log({'threshold': threshold})

        data = [x for x in zip(fpr, tpr)]
        table = logger.wandb.Table(columns=["fpr", "tpr"], data=data)
        roc = logger.wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "fpr", "y": "tpr"},
            {
                "title": "ROC",
                "x-axis-title": "False positive rate",
                "y-axis-title": "True positive rate",
            },
        )

        logger.log({'roc': roc})

        link1, link2 = [], []
        for idx, link in enumerate(path_list):
            if idx % 2 == 0:
                link1.append(link)
            else:
                link2.append(link)
        data = [(l1, l2, p, n) for l1, l2, p, n in zip(link1, link2, fp, fn)]
        table = logger.wandb.Table(columns=["link1", "link2", "fp", "fn"], data=data)
        logger.log({"fp and fn": table})

        logger.log({"conf_mat" : logger.wandb.plot.confusion_matrix(probs=None,
                        y_true=issame_list, preds=predict,
                        class_names=["different person", "same person"])})

        logger.finish_run()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='validate_lfw.py')
    parser.add_argument('--data', type=str, default='./data/lfw', help='dataset')
    parser.add_argument('--pairs-path', type=str, default='./data/pairs.txt', help='pair file to evaluate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size dataloader')

    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')

    parser.add_argument('--detector', type=str, default='retinaface', help='Name detector detect face')
    parser.add_argument('--extractor', type=str, default='facenet', help='Name extractor extract feature face')

    parser.add_argument('--distance_metric', type=int, help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--subtract_mean', action='store_true', help='Subtract feature mean before calculating distance.')
    parser.add_argument('--flip', action='store_true', help='Flip image and add')
    opt = parser.parse_args()
    print(opt)

    attack_lfw(opt)