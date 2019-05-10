import os
import numpy as np
import torch
import glob
import torch.utils.data as Data
from xml.dom.minidom import parse
from utils.data_aug import *


# dataset
class CustomDataset(Data.Dataset):
    def __len__(self):
        return len(self._img_frames)

    def __init__(self, image_frames, label_set, img_size=(512, 512), transform=False):
        super(CustomDataset, self).__init__()
        self._img_frames = image_frames
        self._label_set = label_set
        self._img_size = img_size
        self._transform = transform
        self._data_aug = DataAug()

    # parse the xml file
    def label_parser(self, label_path, pad):
        dom_tree = parse(label_path).documentElement
        objects = dom_tree.getElementsByTagName("object")
        bounding_boxes = []
        for obj in objects:
            name = obj.getElementsByTagName("name")[0].childNodes[0].data
            xmin = int(obj.getElementsByTagName("xmin")[0].childNodes[0].data) + pad[0]
            ymin = int(obj.getElementsByTagName("ymin")[0].childNodes[0].data) + pad[2]
            xmax = int(obj.getElementsByTagName("xmax")[0].childNodes[0].data) + pad[0]
            ymax = int(obj.getElementsByTagName("ymax")[0].childNodes[0].data) + pad[2]
            bbox = {'name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            bounding_boxes.append(bbox)
        return bounding_boxes

    def data_transform(self, img, bboxes_list):
        bboxes = np.zeros(shape=(len(bboxes_list), 5)).astype(np.float32)
        for i in range(len(bboxes_list)):
            x = ((bboxes_list[i]['xmin'] + bboxes_list[i]['xmax']) / 2) / img.size[0]
            y = ((bboxes_list[i]['ymin'] + bboxes_list[i]['ymax']) / 2) / img.size[1]
            w = (bboxes_list[i]['xmax'] - bboxes_list[i]['xmin']) / img.size[0]
            h = (bboxes_list[i]['ymax'] - bboxes_list[i]['ymin']) / img.size[1]
            bboxes[i] = [0 if bboxes_list[i]['name'] == "xiangla" else 1, x, y, w, h]
        bboxes = torch.from_numpy(bboxes)
        img = np.array(img).astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, bboxes


    def __getitem__(self, index):
        img_path = self._img_frames[index]
        img = Image.open(img_path).convert('RGB')
        img, pad = pad_to_square(img)
        label_path = self._label_set[index]
        bboxes = self.label_parser(label_path, pad)
        # resize the image to fixed size
        img, bboxes = self._data_aug.img_resize(img, bboxes, self._img_size)
        # random transform for data augmentation
        if self._transform:
            img, bboxes = self._data_aug.transform(img, bboxes)

        return self.data_transform(img, bboxes)

# customize the output batch data of CustomDataset
def my_collate(batch):
    data = torch.stack([item[0] for item in batch], 0)
    target = [item[1] for item in batch]
    return (data, target)


class MyDataLoader(object):
    def __init__(self, img_path, label_path, img_size=(512, 512), split_ratio=0.05, transform=False):
        img_frames = sorted(glob.glob(os.path.join(img_path, '*')))
        label_set = sorted(glob.glob(os.path.join(label_path, '*')))
        train_data, val_data = self.split_dataset(img_frames, label_set, split_ratio)
        self._train_dataset = CustomDataset(train_data['image'], train_data['label'], img_size, transform=True)
        self._val_dataset = CustomDataset(val_data['image'], val_data['label'], img_size)

    def split_dataset(self, img_frames, label_set, split_ratio):
        total_len = len(img_frames)
        val_len = round(total_len * split_ratio)
        train_img_frames = img_frames[:-val_len]
        train_label_set = label_set[:-val_len]
        val_img_frames = img_frames[-val_len:]
        val_label_set = label_set[-val_len:]
        return {"image": train_img_frames, "label": train_label_set}, {"image": val_img_frames, "label": val_label_set}

    def get_train_dataloader(self, batch_size=4, shuffle=True, num_works=4):
        return Data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_works, collate_fn=my_collate)

    def get_val_dataloader(self, batch_size=4, num_works=4):
        return Data.DataLoader(self._val_dataset, batch_size=batch_size, num_workers=num_works, collate_fn=my_collate)




# from PIL import ImageDraw
#
# image_frames = sorted(glob.glob('../data/img/*'))
# label_set = sorted(glob.glob('../data/label/*'))
#
# dataset = CustomDataset(image_frames, label_set)
# i = 0
# for (img, bboxes) in dataset:
#     if i == 0:
#         print(bboxes)
#         draw = ImageDraw.Draw(img)
#         for i in range(len(bboxes)):
#             draw.line([(bboxes[i]['xmin'], bboxes[i]['ymin']), (bboxes[i]['xmin'], bboxes[i]['ymax'])], fill=(0, 255, 0), width=2)
#             draw.line([(bboxes[i]['xmin'], bboxes[i]['ymax']), (bboxes[i]['xmax'], bboxes[i]['ymax'])], fill=(0, 255, 0), width=2)
#             draw.line([(bboxes[i]['xmax'], bboxes[i]['ymax']), (bboxes[i]['xmax'], bboxes[i]['ymin'])], fill=(0, 255, 0), width=2)
#             draw.line([(bboxes[i]['xmax'], bboxes[i]['ymin']), (bboxes[i]['xmin'], bboxes[i]['ymin'])], fill=(0, 255, 0), width=2)
#         img.show()
#         break
#     i += 1







