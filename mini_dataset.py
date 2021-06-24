import argparse

import os
from tqdm import tqdm
from shutil import copy2

def mini_dataset(opt): 
    for folder in tqdm(os.listdir(opt.data)):
        if os.path.isfile(folder): continue

        cnt = 0
        for filename in os.listdir(os.path.join(opt.data, folder)):
            if cnt == opt.count: break
            
            cnt += 1
            file = os.path.join(opt.data, folder, filename)
            nfile = os.path.join(opt.save_dir, folder, filename)
            os.makedirs(os.path.dirname(nfile), exist_ok=True)
            copy2(file, nfile)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='select_CASIAWebFace.py')
    parser.add_argument('--data', type=str, default='./data/CASIA-WebFace', help='dataset')
    parser.add_argument('--save-dir', type=str, default='./data/CASIA-WebFace-mini', help='Dir save WebFace')
    parser.add_argument('--count', type=int, default=1, help='Amount image per class selectsa')
    opt = parser.parse_args()
    print(opt)

    mini_dataset(opt)