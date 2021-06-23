import torch
import torch.nn as nn

from models.retinaface.layers.modules.multibox_loss import MultiBoxLoss
from models.retinaface.layers.functions.prior_box import PriorBox

class DetectionLoss(nn.Module):
    def __init__(self, cfg, image_size):
        super(DetectionLoss, self).__init__()
        self.cfg = cfg
        self.multiboxloss = MultiBoxLoss(2, 0.45, True, 0, True, 7, 0.35, False)
        self.priorbox = PriorBox(cfg, image_size)
        with torch.no_grad():
            self.priorbox = self.priorbox.forward()

    def forward(self, predictions, targets):
        self.priorbox = self.priorbox.to(predictions[0].device)
        loss_l, loss_c, _ = self.multiboxloss(predictions, self.priorbox, targets)
        return  self.cfg['loc_weight'] * loss_l + loss_c #+ loss_landm