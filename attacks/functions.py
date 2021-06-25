from utils.log import WandbLogger
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn import MSELoss, L1Loss

from utils.general import bboxes2masks, predict2target, time_synchronized, target2bboxes
from utils.image import blur_bboxes
from utils.metrics import psnr, cosine
from models.facemodel import FaceRecognition, FaceVerification

from .optim import get_optim
from .loss import DetectionLoss

@torch.no_grad()
def _attack_face(model, img, bboxes_target, faces_target, mask, loss_detect_fn, loss_face_fn, opt):
    att_img = img.clone()
    att_img.requires_grad = True

    t_bboxes = target2bboxes(bboxes_target, img.shape[-2], img.shape[-1])
    optim = get_optim(opt, [att_img])

    for _ in range(25):
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            out_dectect = model.facedetector(att_img)
            loss_detect = loss_detect_fn(out_dectect, bboxes_target)
            p_bboxes, _ = model.facedetector.get_faces(out_dectect, att_img.shape)

            faces = []
            for idx, (p_boxes, t_boxes) in enumerate(zip(p_bboxes, t_bboxes)):
                boxes = t_boxes if len(p_boxes) != len(t_boxes) else p_boxes

                for box in boxes:
                    face = att_img[idx:idx + 1, :, box[1]:box[3], box[0]:box[2]]
                    face = nn.functional.interpolate(face, size=(160, 160))
                    faces.append(face.squeeze())
                if len(boxes) == 0:
                    face = att_img[idx:idx + 1]
                    face = nn.functional.interpolate(face, size=(160, 160))
                    faces.append(face.squeeze())

            if len(faces) != 0:
                faces = torch.stack(faces)
                out = model.facerecognition(faces)
                loss_face = loss_face_fn(out, faces_target)
            else: 
                loss_face = 0
                
            loss = loss_detect + loss_face

        loss.backward()
        att_img.grad[mask] = 0
        optim.step()
    
    return att_img

@torch.no_grad()
def attack_face(model, img, target, loss_detect_fn, loss_face_fn, logger, opt, delta=False):
    t = time_synchronized()
    (t_bboxes, t_landmarks), names = model(img)

    height, width = img.shape[-2:]
    bboxes_target = predict2target(t_bboxes, t_landmarks, width, height, img.device)
    faces_target = target if target else names

    mask = bboxes2masks(t_bboxes, img.shape, 0.2)
    blur_img = blur_bboxes(img, t_bboxes, opt.kernel_blur, opt.type_blur)
    
    time_preprocess = time_synchronized() - t
    logger.increase_log({"time/preprocess": time_preprocess}) if logger else None

    t = time_synchronized()
    att_img = _attack_face(model, blur_img, bboxes_target, faces_target, mask, loss_detect_fn, loss_face_fn, opt)
    
    logger.increase_log({"time/attack": time_synchronized() - t}) if logger else None

    return att_img, (blur_img - img, att_img - blur_img) if delta else att_img    

@torch.no_grad()
def attack_facerecognition(model:FaceRecognition, img, target, logger:WandbLogger, opt, delta=False):
    loss_detect_fn = DetectionLoss(model.facedetector.cfg, img.shape[-2:]).to(img.device)

    return attack_face(model, img, target, loss_detect_fn, cross_entropy, logger, opt, delta)

@torch.no_grad()
def attack_faceverification(model:FaceVerification, img, target, logger:WandbLogger, opt, delta=False):
    loss_detect_fn = DetectionLoss(model.facedetector.cfg, img.shape[-2:]).to(img.device)
    
    return attack_face(model, img, target, loss_detect_fn, L1Loss(reduction='sum'), logger, opt, delta)