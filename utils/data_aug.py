from PIL import Image, ImageFilter
import random
import math
import numpy as np
import torch.nn.functional as F

def pad_to_square(img):
    w, h = img.size[0], img.size[1]
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    paste_pos = (0, pad1, w, h + pad1) if h <= w else (pad1, 0, w + pad1, h)
    canva = Image.new('RGB', (max(h, w), max(h, w)), (0, 0, 0))
    canva.paste(img, paste_pos)
    return canva, pad


# data augmentation
class DataAug():
    def __init__(self, random_flip=0.1, random_crop=0.3, random_rotate=0.2, gaussian_blur=0.3):
        self._random_flip = random_flip
        self._random_crop = random_crop
        self._random_rotate = random_rotate
        self._gaussian_blur = gaussian_blur

    def img_flip(self, img, bboxes, mode='vertical'):
        _img = None
        if mode == 'horizontal':
            _img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for i in range(len(bboxes)):
                tmp_xmin = bboxes[i]['xmin']
                bboxes[i]['xmin'] = img.size[0] - bboxes[i]['xmax']
                bboxes[i]['xmax'] = img.size[0] - tmp_xmin
        elif mode == 'vertical':
            _img = img.transpose(Image.FLIP_TOP_BOTTOM)
            for i in range(len(bboxes)):
                tmp_ymin = bboxes[i]['ymin']
                bboxes[i]['ymin'] = img.size[0] - bboxes[i]['ymax']
                bboxes[i]['ymax'] = img.size[0] - tmp_ymin
        else:
            raise Exception("Invalid flip parameter!")
        return _img, bboxes

    def img_rotate(self, img, bboxes, degree=90):
        if degree % 90 != 0:
            raise Exception("Invalid rotate degree!")
        _img = img.rotate(degree)
        for i in range(len(bboxes)):
            tmp_ymin = bboxes[i]['ymin']
            tmp_xmin = bboxes[i]['xmin']
            if degree == 90:
                bboxes[i]['xmin'] = bboxes[i]['ymin']
                bboxes[i]['ymin'] = img.size[0] - bboxes[i]['xmax']
                bboxes[i]['xmax'] = bboxes[i]['ymax']
                bboxes[i]['ymax'] = img.size[0] - tmp_xmin
            elif degree == 180:
                bboxes[i]['xmin'] = img.size[0] - bboxes[i]['xmax']
                bboxes[i]['ymin'] = img.size[1] - bboxes[i]['ymax']
                bboxes[i]['xmax'] = img.size[0] - tmp_xmin
                bboxes[i]['ymax'] = img.size[1] - tmp_ymin
            elif degree == 270:
                bboxes[i]['xmin'] = img.size[1] - bboxes[i]['ymax']
                bboxes[i]['ymin'] = tmp_xmin
                bboxes[i]['ymax'] = bboxes[i]['xmax']
                bboxes[i]['xmax'] = img.size[1] - tmp_ymin

        return _img, bboxes

    def img_resize(self, img, bboxes, target_size=(512, 512)):
        _img = img.resize(target_size)
        x_zoom_scale = target_size[0] / img.size[0]
        y_zoom_scale = target_size[1] / img.size[1]
        for i in range(len(bboxes)):
            bboxes[i]['xmin'] *= x_zoom_scale
            bboxes[i]['ymin'] *= y_zoom_scale
            bboxes[i]['xmax'] *= x_zoom_scale
            bboxes[i]['ymax'] *= y_zoom_scale
        return _img, bboxes

    def img_crop(self, img, bboxes, target_size=(512, 512), scale=(0.6, 1.0), ratio=(4. / 5., 5. / 4.)):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)
        w = min(int(round(math.sqrt(target_area * aspect_ratio))), img.size[0])
        h = min(int(round(math.sqrt(target_area / aspect_ratio))), img.size[1])
        axis_x = round(random.uniform(0, img.size[0] - w))
        axis_y = round(random.uniform(0, img.size[1] - h))
        box = (axis_x, axis_y, axis_x + w, axis_y + h)
        _img = img.crop(box)
        for i in range(len(bboxes)):
            bboxes[i]['xmin'] -= axis_x
            bboxes[i]['xmax'] -= axis_x
            bboxes[i]['ymin'] -= axis_y
            bboxes[i]['ymax'] -= axis_y
        return self.img_resize(_img, bboxes, target_size)

    def img_gaussian_blur(self, img):
        _img = img.filter(ImageFilter.GaussianBlur)
        return _img

    def transform(self, img, bboxes):
        if random.random() < self._random_flip:
            img, bboxes = self.img_flip(img, bboxes)
        if random.random() < self._random_crop:
            img, bboxes = self.img_crop(img, bboxes)
        if random.random() < self._random_rotate:
            img, bboxes = self.img_rotate(img, bboxes, random.randint(1, 3) * 90)
        if random.random() < self._gaussian_blur:
            img = self.img_gaussian_blur(img)
        return img, bboxes