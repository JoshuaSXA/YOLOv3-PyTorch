from model.yolo import *
from model.loss import *
from utils.data import *
#
#
# def train_net(net, epochs=10, batch_size=4,lr=0.1, val_percent=0.05, save_cp=True):
#
#     # dataloader
#     dataloader = MyDataLoader("data/img/", "data/label")
#     train_loader = dataloader.get_train_dataloader(batch_size=4, shuffle=True, num_works=0)
#     val_loader = dataloader.get_val_dataloader(batch_size=4, num_works=0)
#
#     # train
#     for epoch in range(epochs):
#         epoch_train_loss = 0.0
#         # net.train()
#         for (img, label) in train_loader:
#             print(label[0].size(), label[1].size(), label[2].size(), label[3].size())
#             break
#         break
#
#
# if __name__ == '__main__':
#     net = YOLO(input_channels=3, anchor_num=3, class_num=2)
#     train_net(net)


# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter


config = {
    "data":{
      "img_path":"data/img/",
      "label_path":"data/label/",
    },
    "img_size":(512, 512),
    "img_channels":3,
    "anchor_num":3,
    "anchors":[[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]],
    "class_num":2,
    "optimizer":{
        "type":"adam",
        "weight_decay":4e-5
    },
    "lr": {
            "value": 0.01,
            "freeze_backbone": False,
            "decay_gamma": 0.1,
            "decay_step": 20,
    },
    "batch_size":4,
    "epochs":100,
    "model_save_dir":"weights/"
}


def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    # Load and initialize network
    net = YOLO(input_channels=config['img_channels'], anchor_num=config['anchor_num'], class_num=config['class_num'])
    net.train()
    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda() if torch.cuda.is_available() else net

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["lr"]["decay_step"], gamma=config["lr"]["decay_gamma"])

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["anchors"][i], config["class_num"], config['img_size']))

    # DataLoader
    dataloader = MyDataLoader(config['data']['img_path'], config['data']['label_path'], config['img_size'], 0.05, True)
    train_loader = dataloader.get_train_dataloader(config['batch_size'], num_works=4, shuffle=True)

    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        step = 1
        for (images, label) in train_loader:
            # Forward and backward
            # print(label)
            optimizer.zero_grad()
            outputs = net(images)
            # print(outputs[0].size(), outputs[1].size(), outputs[2].size())
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], label)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            logging.info(
                "epoch [%.3d] iter = %d loss = %.2f lr = %.5f " %
                (epoch, step, loss.item(), lr)
            )
            step += 1
        lr_scheduler.step()
        _save_checkpoint(net.state_dict(), config['model_save_dir'], epoch)

    logging.info("Training finished!")


# best_eval_result = 0.0
def _save_checkpoint(state_dict, model_save_dir, epoch_num, evaluate_func=None):
    # global best_eval_result
    model_name = "model_" + str(epoch_num) + ".pth"
    checkpoint_path = os.path.join(model_save_dir, model_name)
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def _get_optimizer(config, net):
    optimizer = None
    params = net.parameters()
    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"], amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))
    return optimizer


if __name__ == "__main__":
    train(config)