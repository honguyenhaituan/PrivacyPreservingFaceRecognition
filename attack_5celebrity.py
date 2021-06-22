import os
import torch
import numpy as np
from tqdm import tqdm

import argparse
from pathlib import Path
from threading import Thread
from utils.log import WandbLogger

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from models.FaceRecogniton import *
from attacks.functions import attack_facerecognition
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.general import increment_path

def attack_5celebrity(opt):
    logger = WandbLogger("PrivacyPreservingFaceRecognition-5celebrity", None, opt)
    logger_attack_img, logger_compare_img = [], []
    save_dir = increment_path(Path(opt.save_dir) / "exp", exist_ok=False)  # increment run
    (save_dir / 'attack_img' if opt.save_attack_image else save_dir).mkdir(parents=True, exist_ok=True)
    (save_dir / 'compare_img' if opt.save_compare_image else save_dir).mkdir(parents=True, exist_ok=True)


    workers = 0 if os.name == 'nt' else 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data, x),
                                          transform=transforms.ToTensor())
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], num_workers=workers)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    class_names = {k:v for k,v in enumerate(class_names)}
    class_set = logger.wandb.Classes([
        {'name': name, 'id': int(id)} 
        for id, name in class_names.items()
    ])

    retinaface = retinaface_mnet(pretrained=True).eval()
    facenet = InceptionResnetV1(classify=True, num_classes=len(class_names)).eval()
    facenet.load_state_dict(torch.load(opt.pretrain_facenet))
    facerecognition = FaceRecognition(retinaface, facenet).to(device)

    pred, label = [], []
    for image, target in dataloaders["val"]:
        image = image.to(device)
        att_img, (delta_blur, delta_att) = attack_facerecognition(facerecognition, image, logger, opt, delta=True)
        (bboxes, _), name = facerecognition(att_img)

        for _pred, _label in zip(name, target):
            pred.append(_pred.item())
            label.append(_label.item())

        if opt.save_attack_image: 
            for _img, _target, _bboxes, _name, in zip(att_img, target, bboxes, [name]):
                box_data = [{"position": {"minX": box[0], "minY": box[1], "maxX": box[2], "maxY": box[3]},
                                    "class_id": int(cls),
                                    "box_caption": class_names[cls],
                                    "domain": "pixel"} for box, cls in zip(_bboxes.tolist(), _name.tolist())]
                boxes = {"predictions": {"box_data": box_data, "class_labels": class_names}}  # inference-space
                logger_attack_img.append([len(logger_attack_img), logger.wandb.Image(_img, boxes=boxes, caption=class_names[_target.item()], classes=class_set), class_names[_target.item()]])

                save_image(_img, os.path.join(save_dir, "attack_img", "%i.png" % len(logger_attack_img)))

        if opt.save_compare_image:
            for _img, _att_img, _delta_blur, _delta_att in zip(image, att_img, delta_blur, delta_att):
                image_compare = make_grid([_img, _delta_blur * 0.5 + 0.5, _delta_att * 0.5 + 0.5, _att_img])
                save_image(image_compare, os.path.join(save_dir, "compare_img", "%i.png" % len(pred)))
                logger_compare_img.append(logger.wandb.Image(os.path.join(save_dir, "compare_img", "%i.png" % len(pred))))

    logger.log({"metrics/accuracy": accuracy_score(label, pred)})
    logger.log({"metrics/f1": f1_score(label, pred, average="micro")})
    logger.log({"metrics/precision": precision_score(label, pred, average="micro")})
    logger.log({"metrics/recall": recall_score(label, pred, average="micro")})

    if logger_attack_img:
        columns = ["id", "image", "ground truth"]
        table = logger.wandb.Table(data=logger_attack_img, columns=columns)
        logger.log({"Bounding Box Debugger/Images": table})

    if logger_compare_img:
        logger.log({"Comapare image": logger_compare_img})

    logger.finish_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='attack_5celebrity.py')
    parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
    parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')

    parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
    parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

    parser.add_argument('--data', type=str, default='/content/drive/MyDrive/data/5 Celebrity Faces Dataset', help='dataset')
    parser.add_argument('--pretrain-facenet', type=str, default='/content/drive/MyDrive/pretrain/face_recognition.pth', help='Path pretrain')
    
    parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
    parser.add_argument('--save-attack-image', action='store_true', help='Save image file after attack')    
    parser.add_argument('--save-compare-image', action='store_true', help='Save original, delta and attack image')    
        
    opt = parser.parse_args()
    print(opt)

    attack_5celebrity(opt)