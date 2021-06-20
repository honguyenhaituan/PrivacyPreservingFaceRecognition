from utils.log import WandbLogger
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from utils.general import bboxes2masks, predict2target, time_synchronized
from utils.image import blur_bboxes, pixelate_bboxes
from utils.metrics import psnr, cosine
from models.FaceRecogniton import FaceRecognition
from models.retinaface.layers.modules.multibox_loss import MultiBoxLoss
from models.retinaface.layers.functions.prior_box import PriorBox

class I_FGSM: 
    def __init__(self, params, epsilon=20): 
        self.params = params
        self.epsilon = epsilon / 255
        self.alpha = 1/ 255
        self.updated_params = []
        for param in self.params:
            self.updated_params.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        return -self.alpha * torch.sign(self.params[idx].grad)

    @torch.no_grad()
    def step(self):
        for idx, (param, updated_param) in enumerate(zip(self.params, self.updated_params)):
            if param is None: 
                continue
    
            n_update = torch.clamp(updated_param + self._cal_update(idx), -self.epsilon, self.epsilon)
            update = n_update - updated_param
            n_param = torch.clamp(param + update, 0, 1)
            update = n_param - param

            param += update
            updated_param += update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class MI_FGSM(I_FGSM):
    def __init__(self, params, epsilon=20, momemtum=0):
        super(MI_FGSM, self).__init__(params, epsilon)
        self.momentum = momemtum
        self.o_grad = []
        for param in self.params:
            self.o_grad.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        grad = self.o_grad[idx] * self.momentum + self.params[idx].grad / torch.sum(torch.abs(self.params[idx].grad))
        return -self.alpha * torch.sign(grad)

    def zero_grad(self):
        for o_grad, param in zip(self.o_grad, self.params):
            if param.grad is not None:
                o_grad = o_grad * self.momentum + param.grad / torch.sum(torch.abs(param.grad))
        super().zero_grad()

def get_method_attack(name_attack, params, epsilon, momentum) -> I_FGSM:
    if name_attack == 'I-FGSM': 
        return I_FGSM(params, epsilon)
    if name_attack == 'MI-FGSM':
        return MI_FGSM(params, epsilon, momentum)
    return None

class DetectionLoss(nn.Module):
    def __init__(self, cfg, image_size):
        super(DetectionLoss, self).__init__()
        self.cfg = cfg
        self.multiboxloss = MultiBoxLoss(2, 0.45, True, 0, True, 7, 0.35, False)
        self.priorbox = PriorBox(cfg, image_size)
        with torch.no_grad():
            self.priorbox = self.priorbox.forward()

    def forward(self, predictions, targets):
        self.priorbox = self.priorbox.to(targets.device)
        loss_l, loss_c, _ = self.multiboxloss(predictions, self.priorbox, targets)
        return  self.cfg['loc_weight'] * loss_l + loss_c #+ loss_landm

@torch.no_grad()
def attack_facerecognition(model:FaceRecognition, img, name_attack, epsilon, momentum, logger:WandbLogger):
    t = time_synchronized()
    (bboxes, landmarks), names = model(img)

    height, width = img.shape[-2:]
    bboxes_target = predict2target(bboxes, landmarks, width, height)

    mask = bboxes2masks(bboxes, img.shape, 0.2)

    att_img = pixelate_bboxes(img, bboxes)
    att_img.requires_grad = True

    attack = get_method_attack(name_attack, [att_img], epsilon, momentum)
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

    return att_img