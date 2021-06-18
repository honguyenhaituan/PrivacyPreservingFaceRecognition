from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def blur_bboxes(images, bboxes, kernel=9):
    gaussian_blur = transforms.GaussianBlur(kernel)
    result = images.clone()
    for image, boxes in zip(result, bboxes):
        for box in boxes: 
            crop_bbox = image[:, box[1]:box[3], box[0]:box[2]]
            image[:, box[1]:box[3], box[0]:box[2]] = gaussian_blur(crop_bbox)

    return result

def _pixelate(image, kernel):
    h, w = image.shape[-2:]
    dh, dw = (h + kernel - 1) // kernel, (w + kernel - 1) // kernel 

    temp = transforms.Resize((dh, dw))(image)
    return transforms.Resize((h,w), InterpolationMode.NEAREST)(temp)

def pixelate_bboxes(images, bboxes, kernel=9):
    result = images.clone()
    for image, boxes in zip(result, bboxes):
        for box in boxes: 
            crop_bbox = image[:, box[1]:box[3], box[0]:box[2]]
            image[:, box[1]:box[3], box[0]:box[2]] = _pixelate(crop_bbox, kernel)

    return result