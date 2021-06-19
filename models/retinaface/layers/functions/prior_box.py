import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_size = self.min_sizes[k]
            mat = np.array(list(product(range(f[0]), range(f[1]), min_size))).astype(np.float32)
            mat[:, 0], mat[:, 1] = ((mat[:, 1] + 0.5) * self.steps[k] / self.image_size[1],
                                    (mat[:, 0] + 0.5) * self.steps[k] / self.image_size[0])
            mat = np.concatenate([mat, mat[:, 2:3]], axis=1)
            mat[:, 2] = mat[:, 2] / self.image_size[1]
            mat[:, 3] = mat[:, 3] / self.image_size[0]
            anchors.append(mat)
        output = np.concatenate(anchors, axis=0)
        if self.clip:
            output = np.clip(output, 0, 1)
        return torch.from_numpy(output)