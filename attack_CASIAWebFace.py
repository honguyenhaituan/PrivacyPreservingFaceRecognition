import os
import torch

import argparse
from pathlib import Path
from utils.log import WandbLogger
from tqdm import tqdm

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

def attack_CASIAWebFace(opt):
    logger = WandbLogger("PrivacyPreservingFaceRecognition-CASIAWebFace", None, opt) if opt.log_wandb else None
    save_dir = increment_path(Path(opt.save_dir), exist_ok=False)  # increment run
    save_image_dir = str(save_dir / 'attack')
    save_compare_dir = str(save_dir / 'compare')
    
    dataset = ImageFolderWithPaths(opt.data, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=workers)
    facerecognition = facerecognition_retinaface_facenet(pretrained='casia-webface').eval().to(device)

    preds_att, preds_img, label = [], [], []
    for image, target, paths in tqdm(dataloader):
        image = image.to(device)
        _, pred_img = facerecognition(image)

        target_attack = target.to(device) if opt.label_target else None
        att_img, (delta_blur, delta_att) = attack_facerecognition(facerecognition, image, target_attack, logger, opt, delta=True)
        (bboxes, _), pred_att = facerecognition(att_img)

        if opt.save_attack_image: 
            logger_attack_img = []
            start = 0
            for _img, _target, _bboxes, _name, path in zip(att_img, target, bboxes, pred_att, paths):
                save_path = path.replace(opt.data, save_image_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image(_img, save_path)

                if opt.log_wandb:
                    box_data = [{"position": {"minX": box[0], "minY": box[1], "maxX": box[2], "maxY": box[3]},
                                        "class_id": int(cls),
                                        "box_caption": str(cls),
                                        "domain": "pixel"} for box, cls in zip(_bboxes.tolist(), pred_att[start:start+len(_bboxes)].tolist())]
                    boxes = {"predictions": {"box_data": box_data}} #, "class_labels": class_names}}  # inference-space
                    logger_attack_img.append(logger.wandb.Image(_img, boxes=boxes, caption="label: " + str(_target)))
                    start += len(_bboxes)

            if opt.log_wandb: logger.log({"Predict image": logger_attack_img})

        if opt.save_compare_image:
            logger_compare_img = []
            for _img, _att_img, _delta_blur, _delta_att, path in zip(image, att_img, delta_blur, delta_att, paths):
                save_path = path.replace(opt.data, save_compare_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                image_compare = make_grid([_img, _delta_blur * 0.5 + 0.5, _delta_att * 0.5 + 0.5, _att_img])
                save_image(image_compare, save_path)
                if opt.log_wandb: logger_compare_img.append(logger.wandb.Image(save_path))

            if opt.log_wandb: logger.log({"Comapare image": logger_compare_img})

        preds_att.extend(pred_att.to('cpu').numpy())
        preds_img.extend(pred_img.to('cpu').numpy())
        label.extend(target.numpy())

        acc_att = accuracy_score(label, preds_att)
        acc_pred = accuracy_score(label, preds_img)
        acc_att_pred = accuracy_score(preds_img, preds_att)
        print("ACC att, pred, att_pred: ", acc_att, acc_pred, acc_att_pred)
        
        if opt.log_wandb:
            logger.log({"metrics/accuracy": accuracy_score(label, preds_att)})
            logger.log({"metrics/f1": f1_score(label, preds_att, average="micro")})
            logger.log({"metrics/precision": precision_score(label, preds_att, average="micro")})
            logger.log({"metrics/recall": recall_score(label, preds_att, average="micro")})
            logger.end_epoch()

    if logger: logger.finish_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='attack_CASIAWebFace.py')
    parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
    parser.add_argument('--max_iter', type=int, default=25, help='Max iter loop to process attack')
    parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')
    parser.add_argument('--label-target', action='store_true', help='Use ground truth to attack model')


    parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
    parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

    parser.add_argument('--data', type=str, default='./data/CASIA-WebFace-mini', help='dataset')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size dataloader')
    
    parser.add_argument('--save-dir', type=str, default='./data/CASIA-WebFace-mini-attack', help='Dir save all result')
    parser.add_argument('--save-attack-image', action='store_true', help='Save image file after attack')    
    parser.add_argument('--save-compare-image', action='store_true', help='Save original, delta and attack image')    
        
    parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')

    opt = parser.parse_args()
    print(opt)

    attack_CASIAWebFace(opt)