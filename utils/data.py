import os
import numpy as np
from PIL import Image, ImageFilter
import torch
import random
import math
import glob
from torchvision import transforms
import torch.utils.data as Data
from xml.dom.minidom import parse


# data augmentation
class DataAug():
    def __init__(self, random_flip=0.5, random_crop=0.3, random_rotate=0.2, gaussian_blur=0.2):
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
                bboxes[i]['xmin'] = img.size[1] - bboxes[i]['ymax']
                bboxes[i]['ymin'] = tmp_xmin
                bboxes[i]['ymax'] = bboxes[i]['xmax']
                bboxes[i]['xmax'] = img.size[1] - tmp_ymin
            elif degree == 180:
                bboxes[i]['xmin'] = img.size[0] - bboxes[i]['xmax']
                bboxes[i]['ymin'] = img.size[1] - bboxes[i]['ymax']
                bboxes[i]['xmax'] = img.size[0] - tmp_xmin
                bboxes[i]['ymax'] = img.size[1] - tmp_ymin
            elif degree == 270:
                bboxes[i]['xmin'] = bboxes[i]['ymin']
                bboxes[i]['ymin'] = img.size[0] - bboxes[i]['xmax']
                bboxes[i]['xmax'] = bboxes['ymax']
                bboxes[i]['ymax'] = img.size[0] - tmp_xmin
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
            bboxes[i]['xmin'] += axis_x
            bboxes[i]['xmax'] += axis_x
            bboxes[i]['ymin'] += axis_y
            bboxes[i]['ymax'] += axis_y
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


# dataset
class CustomDataset(Data.Dataset):
    def __len__(self):
        return len(self._img_frames)

    def __init__(self, image_frames, label_set, transform=False):
        super(CustomDataset, self).__init__()
        self._img_frames = image_frames
        self._label_set = label_set
        self._transform = transform
        self._data_aug = DataAug()

    # parse the xml file
    def label_parser(self, label_path):
        dom_tree = parse(label_path).documentElement
        objects = dom_tree.getElementsByTagName("object")
        bounding_boxes = []
        for obj in objects:
            name = obj.getElementsByTagName("name")[0].childNodes[0].data
            xmin = int(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
            bbox = {'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            bounding_boxes.append(bbox)
        return bounding_boxes

    def data_transform(self, img, bboxes_list):
        img = np.array(img).astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        bboxes = np.zeros(shape=(len(bboxes_list), 5)).astype(np.int32)
        for i in range(len(bboxes_list)):
            bboxes[i] = [0 if bboxes_list[i]['name'] == "xiangla" else 1, bboxes_list[i]['xmin'],
                         bboxes_list[i]['ymin'], bboxes_list[i]['xmax'], bboxes_list[i]['ymax']]
        bboxes = torch.from_numpy(bboxes)
        return img, bboxes


    def __getitem__(self, index):
        img_path = self._img_frames[index]
        img = Image.open(img_path)
        label_path = self._label_set[index]
        bboxes = self.label_parser(label_path)
        # resize the image to fixed size
        img, bboxes = self._data_aug.img_resize(img, bboxes, (512, 512))
        # random transform for data augmentation
        if self._transform:
            img, bboxes = self._data_aug.transform(img, bboxes)
        return self.data_transform(img, bboxes)

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)



class MyDataLoader(object):
    def __init__(self, img_path, label_path, split_ratio=0.05, transform=False):
        img_frames = sorted(glob.glob(os.path.join(img_path, '*')))
        label_set = sorted(glob.glob(os.path.join(label_path, '*')))
        self._split_ratio = split_ratio
        train_data, val_data = self.split_dataset(img_frames, label_set)
        self._train_dataset = CustomDataset(train_data['image'], train_data['label'])
        self._val_dataset = CustomDataset(val_data['image'], val_data['label'])

    def split_dataset(self, img_frames, label_set):
        total_len = len(img_frames)
        val_len = round(total_len * self._split_ratio)
        train_img_frames = img_frames[:-val_len]
        train_label_set = label_set[:-val_len]
        val_img_frames = img_frames[-val_len:]
        val_label_set = label_set[-val_len:]
        return {"image": train_img_frames, "label": train_label_set}, {"image": val_img_frames, "label": val_label_set}

    def get_train_dataloader(self, batch_size=4, shuffle=True, num_works=4):
        return Data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_works, collate_fn=my_collate)

    def get_val_dataloader(self, batch_size=4, num_works=4):
        return Data.DataLoader(self._val_dataset, batch_size=batch_size, num_workers=num_works)




# image_frames = sorted(glob.glob('../data/img/*'))
# label_set = sorted(glob.glob('../data/label/*'))
#
# dataset = CustomDataset(image_frames, label_set)
#
# for (img, bboxes) in dataset:
#     print(bboxes)







