# PrivacyPreservingFaceRecognition
## 0. Cài đặt thư viện
```
pip install -qr requirements.txt
```
## 1. Tập dữ liệu Five celebrity

```
python attack_5celebrity.py
```
Đây là các tham số có thể điều chỉnh khi chạy trên tập Five celebrity
```
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
        
```

Để chạy lại chính xác kết quả của khóa luận, vui lòng tham khảo [Five Celebrity Log Wandb](https://wandb.ai/honguyenhaituan/PrivacyPreservingFaceRecognition-5celebrity)

## 2. Tập dữ liệu CASIA-WEBFACE
```
python attack_CASIAWebFace.py
```
Đây là các tham số có thể điều chỉnh khi chạy trên tập CASIA-WebFace
```
parser.add_argument('--name-attack', type=str, default='I-FGSM', help='name method attack model')
parser.add_argument('--max_iter', type=int, default=20, help='Max iter loop to process attack')
parser.add_argument('--epsilon', type=float, default=15, help='Max value per pixel change')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')
parser.add_argument('--label-target', action='store_true', help='Use ground truth to attack model')


parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

parser.add_argument('--data', type=str, default='./data/CASIA-WebFace-mini', help='dataset')
parser.add_argument('--batch-size', type=int, default=1, help='batch size dataloader')

parser.add_argument('--save-dir', type=str, default='./data/CASIA-WebFace-mini-attack', help='Dir save all result')
parser.add_argument('--save-attack-image', action='store_true', help='Save image file after attack')    
parser.add_argument('--save-compare-image', action='store_true', help='Save original, delta and attack image')    
    
parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')

```
Để chạy lại chính xác kết quả của khóa luận, vui lòng tham khảo [CASIA-WebFace Log Wandb](https://wandb.ai/honguyenhaituan/PrivacyPreservingFaceRecognition-CASIAWebFace)

## 3. Tập dữ liệu LFW
```
python attack_lfw.py
```
Đây là các tham số có thể điều chỉnh khi chạy trên tập LFW
```
parser.add_argument('--name-attack', type=str, default='RMSprop', help='name method attack model')
parser.add_argument('--max_iter', type=int, default=25, help='Max iter loop to process attack')
parser.add_argument('--epsilon', type=float, default=20, help='Max value per pixel change')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum gradient attack')

parser.add_argument('--type-blur', type=int, default=2, help='Choose type blur face image(0: None, 1: gaussian, 2: pixelate)')
parser.add_argument('--kernel-blur', type=int, default=9, help='Kernel of algorithm blur')

parser.add_argument('--data', type=str, default='./data/lfw', help='dataset')
parser.add_argument('--batch-size', type=int, default=32, help='batch size dataloader')

parser.add_argument('--extractor', type=str, default='facenet', help='Name extractor extract feature face')

parser.add_argument('--save-dir', type=str, default='./data/lfw-attack', help='Dir save all result')
parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')
```

Để đánh giá kết quả dữ liệu lfw
```
python validate_lfw.py
```
Đây là các tham số có thể điều chỉnh khi chạy đánh giá
```
parser.add_argument('--data', type=str, default='./data/lfw', help='dataset')
parser.add_argument('--pairs-path', type=str, default='./data/pairs.txt', help='pair file to evaluate')
parser.add_argument('--batch-size', type=int, default=32, help='batch size dataloader')

parser.add_argument('--save-dir', type=str, default='./results', help='Dir save all result')
parser.add_argument('--log-wandb', action='store_true', help='Log something in wandb')

parser.add_argument('--detector', type=str, default='retinaface', help='Name detector detect face')
parser.add_argument('--extractor', type=str, default='facenet', help='Name extractor extract feature face')

parser.add_argument('--distance_metric', type=int, help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
parser.add_argument('--subtract_mean', action='store_true', help='Subtract feature mean before calculating distance.')
parser.add_argument('--flip', action='store_true', help='Flip image and add')
```
Để chạy lại đúng kết quả của khóa luận, vui lòng tham khảo [LFW attack log wandb](https://wandb.ai/honguyenhaituan/PrivacyPreservingFaceRecognition-lfw) và [LFW evaluate log wandb](https://wandb.ai/honguyenhaituan/PrivacyPreservingFaceRecognition-lfw-evaluate)
