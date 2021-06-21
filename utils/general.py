import torch
import time
import glob
import re
from pathlib import Path

def bboxes2masks(bboxes, shape, reduce=0):
    if reduce > 1 or reduce < 0:
        raise ValueError("Reduce must be in 0 to 1")

    masks = torch.ones(shape, dtype=torch.bool)
    for boxes, mask in zip(bboxes, masks): 
        for box in boxes:
            height, width = box[3] - box[1], box[2] - box[0]
            sh, sw = int(height * reduce / 2), int(width * reduce / 2)
            mask[:, box[1] + sh:box[3] - sh, box[0] + sw:box[2] - sw] = 0

    return masks

def predict2target(bboxes, landmarks, width, height):
    target = torch.cat((bboxes[:, :, :-1], landmarks, bboxes[:, :, -1:]), dim=-1)
    target = target.float()
    target[:, :, -1] = 1
    target[:, :, (0, 2)] /= width
    target[:, :, (1, 3)] /= height

    return target
    
def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path