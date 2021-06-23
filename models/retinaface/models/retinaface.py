import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from .net import MobileNetV1 as MobileNetV1
from .net import FPN as FPN
from .net import SSH as SSH

from ..data import cfg_mnet, cfg_re50
from ..utils.loader import load_pretrain
from ..layers.functions.prior_box import PriorBox
from ..utils.box_utils import decode, decode_landm
from ..utils.nms.py_cpu_nms import py_cpu_nms


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(
        self, cfg = None, select_largest=True, keep_all=False, confidence_threshold=0.5):
        """
        :param cfg:  Network related settings.
        """
        super(RetinaFace,self).__init__()
        self.cfg = cfg
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.confidence_threshold = confidence_threshold
        backbone = None

        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("models/retinaface/weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def _transform_input(self, image):
        out = image * 255
        out = out[:, [2, 1, 0]]
        out = transforms.Normalize((104, 117, 123), (1, 1, 1))(out)

        return out

    def forward(self,inputs):
        out = self._transform_input(inputs)
        out = self.body(out)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return (bbox_regressions, classifications, ldm_regressions)

    @torch.no_grad()
    def nms(self, out, img_shape, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        loc, conf, landms = out
        conf = F.softmax(conf, dim=-1)

        device = loc.device
        im_height, im_width = img_shape[2:]
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img_shape[3], img_shape[2], img_shape[3], img_shape[2],
                               img_shape[3], img_shape[2], img_shape[3], img_shape[2],
                               img_shape[3], img_shape[2]]).to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)

        return dets, landms

    @torch.no_grad()
    def get_faces(self, out, img_shape):
        bboxes, confs, lands = out
        r_dets, r_landms = [], []
        for loc, conf, land in zip(bboxes, confs, lands):
            dets, ladms = self.nms((loc, conf, land), img_shape)
            r_det, r_landm = np.array([]), np.array([])
            if self.keep_all:
                r_det, r_landm = dets, ladms
            elif self.select_largest: 
                area = 0
                for idx, (det, ladm) in enumerate(zip(dets, ladms)):
                    if (det[3] - det[1]) * (det[2] - det[0]) > area: 
                        area = (det[3] - det[1]) * (det[2] - det[0])
                        r_det, r_landm = (dets[idx:idx + 1], ladms[idx:idx + 1])
            else:
                r_det, r_landm = (det[0:1], ladm[0:1])  
            if r_det != np.array([]):
                r_det[:, (0, 2)] = np.clip(r_det[:, (0, 2)], 0, img_shape[-1])
                r_det[:, (1, 3)] = np.clip(r_det[:, (1, 3)], 0, img_shape[-2]) 
            r_dets.append(torch.from_numpy(r_det.astype(int)))
            r_landms.append(torch.from_numpy(r_landm.astype(int)))

        return r_dets, r_landms

    @torch.no_grad()
    def detect_faces(self, img):
        out = self.forward(img)
        return self.get_faces(out, img.shape)
        
def retinaface_mnet(pretrained=False):
    model = RetinaFace(cfg_mnet)
    if pretrained:
        model = load_pretrain(model, 'mnet')

    return model

def retinaface_rnet(pretrained=False):
    model = RetinaFace(cfg_re50)
    if pretrained:
        model = load_pretrain(model, 'rnet')

    return model