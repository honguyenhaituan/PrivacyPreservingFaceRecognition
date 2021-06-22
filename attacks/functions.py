from models.FaceVerification import FaceVerification
from utils.log import WandbLogger
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from utils.general import bboxes2masks, predict2target, time_synchronized
from utils.image import blur_bboxes
from utils.metrics import psnr, cosine
from models.FaceRecogniton import FaceRecognition

from .FGSM import get_method_attack
from .loss import DetectionLoss

@torch.no_grad()
def attack_facerecognition(model:FaceRecognition, img, logger:WandbLogger, opt, delta=False):
    t = time_synchronized()
    (bboxes, landmarks), names = model(img)

    height, width = img.shape[-2:]
    bboxes_target = predict2target(bboxes, landmarks, width, height)

    mask = bboxes2masks(bboxes, img.shape, 0.2)

    att_img = blur_bboxes(img, bboxes, opt.kernel_blur, opt.type_blur)
    att_img.requires_grad = True
    if delta: 
        blur_image = att_img.clone()

    attack = get_method_attack(opt, [att_img])
    loss_detect_fn = DetectionLoss(model.facedetector.cfg, img.shape[-2:]).to(att_img.device)
    loss_recog_fn = cross_entropy
    
    time_preprocess = time_synchronized() - t
    logger.increase_log({"time/preprocess": time_preprocess}) if logger else None

    t = time_synchronized()
    for _ in range(100):
        attack.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.facedetector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)

        loss_detect.backward()
        att_img.grad[mask] = 0
        attack.step()

    for _ in range(25):
        attack.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.facedetector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)

            bboxes, _ = model.facedetector.get_faces(out_dectect, att_img.shape)
            bboxes = bboxes.astype(int)

            faces = []
            for idx, boxes in enumerate(bboxes):
                for box in boxes:
                    face = att_img[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                    face = nn.functional.interpolate(face, size=(160, 160))
                    faces.append(face.squeeze())

            faces = torch.stack(faces)

            out = model.facerecognition(faces)
            loss_recog = loss_recog_fn(out, names)
            loss = loss_detect + loss_recog

        loss.backward()
        att_img.grad[mask] = 0
        attack.step()
    
    time_attack = time_synchronized() - t
    logger.increase_log({"time/attack": time_attack}) if logger else None

    if delta: 
        return att_img, (blur_image - img, att_img - blur_image)
    else:
        return att_img

@torch.no_grad()
def attack_facereverification(model:FaceVerification, img, logger:WandbLogger, opt, delta=False):
    t = time_synchronized()
    (bboxes, landmarks), names = model(img)

    height, width = img.shape[-2:]
    bboxes_target = predict2target(bboxes, landmarks, width, height)

    mask = bboxes2masks(bboxes, img.shape, 0.2)

    att_img = blur_bboxes(img, bboxes, opt.kernel_blur, opt.type_blur)
    att_img.requires_grad = True
    if delta: 
        blur_image = att_img.clone()

    attack = get_method_attack(opt, [att_img])
    loss_detect_fn = DetectionLoss(model.facedetector.cfg, img.shape[-2:]).to(att_img.device)
    loss_recog_fn = cross_entropy
    
    time_preprocess = time_synchronized() - t
    logger.increase_log({"time/preprocess": time_preprocess}) if logger else None

    t = time_synchronized()
    for _ in range(100):
        attack.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.facedetector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)

        loss_detect.backward()
        att_img.grad[mask] = 0
        attack.step()

    for _ in range(25):
        attack.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.facedetector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)

            bboxes, _ = model.facedetector.get_faces(out_dectect, att_img.shape)
            bboxes = bboxes.astype(int)

            faces = []
            for idx, boxes in enumerate(bboxes):
                for box in boxes:
                    face = att_img[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                    face = nn.functional.interpolate(face, size=(160, 160))
                    faces.append(face.squeeze())

            faces = torch.stack(faces)

            out = model.facerecognition(faces)
            loss_recog = loss_recog_fn(out, names)
            loss = loss_detect + loss_recog

        loss.backward()
        att_img.grad[mask] = 0
        attack.step()
    
    time_attack = time_synchronized() - t
    logger.increase_log({"time/attack": time_attack}) if logger else None

    if delta: 
        return att_img, (blur_image - img, att_img - blur_image)
    else:
        return att_img