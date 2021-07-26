from os import name
from ..utils.log import WandbLogger
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn import MSELoss, L1Loss

from ..utils.general import bboxes2masks, predict2target, time_synchronized, target2bboxes
from ..utils.image import blur_bboxes
from ..utils.metrics import psnr, cosine
from ..models.facemodel import FaceRecognition, FaceVerification

from .optim import get_optim
from .loss import DetectionLoss

@torch.no_grad()
def _attack_face(model: FaceVerification, img, bboxes_target, faces_target, mask, loss_detect_fn, loss_emb_fn, opt):
    att_img = img.clone()
    att_img.requires_grad = True

    t_bboxes = target2bboxes(bboxes_target, img.shape[-2], img.shape[-1])
    optim = get_optim(opt, [att_img])

    best_loss = float('inf')
    for _ in range(opt.max_iter):
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.detector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)
            p_bboxes, _ = model.detector.detect(out_dectect, isOut=True)

            bboxes = []
            for p_boxes, t_boxes in zip(p_bboxes, t_bboxes):
                boxes = t_boxes if len(p_boxes) != len(t_boxes) else p_boxes
                bboxes.append(boxes)

            out = model.embedding(att_img, bboxes)
            loss_emb = loss_emb_fn(out, faces_target)
            if loss_emb.item() < best_loss: 
                best_loss = loss_emb.item()
                result = att_img.clone()

            loss = loss_detect + loss_emb

        loss.backward()
        att_img.grad[mask] = 0
        optim.step()
    
    return result

@torch.no_grad()
def attack_face(model, img, target, loss_detect_fn, loss_emb_fn, logger, opt, delta=False):
    t = time_synchronized()
    (t_bboxes, t_landmarks), names = model(img)

    height, width = img.shape[-2:]
    bboxes_target = predict2target(t_bboxes, t_landmarks, width, height, img.device)
    if target is None:
        faces_target = names
    else:
        faces_target = target
    # faces_target = target if target else names

    mask = bboxes2masks(t_bboxes, img.shape, 0.05)
    blur_img = blur_bboxes(img, t_bboxes, opt.kernel_blur, opt.type_blur)
    
    time_preprocess = time_synchronized() - t
    logger.increase_log({"time/preprocess": time_preprocess}) if logger else None

    t = time_synchronized()
    att_img = _attack_face(model, blur_img, bboxes_target, faces_target, mask, loss_detect_fn, loss_emb_fn, opt)
    
    logger.increase_log({"time/attack": time_synchronized() - t}) if logger else None

    return (att_img, (blur_img - img, att_img - blur_img)) if delta else att_img    

@torch.no_grad()
def attack_facerecognition(model:FaceRecognition, img, target, logger:WandbLogger, opt, delta=False):
    loss_detect_fn = DetectionLoss(model.detector.cfg, img.shape[-2:]).to(img.device)

    return attack_face(model, img, target, loss_detect_fn, cross_entropy, logger, opt, delta)

@torch.no_grad()
def attack_faceverification(model:FaceVerification, img, target, logger:WandbLogger, opt, delta=False):
    loss_detect_fn = DetectionLoss(model.detector.cfg, img.shape[-2:]).to(img.device)
    
    return attack_face(model, img, target, loss_detect_fn, L1Loss(reduction='sum'), logger, opt, delta)