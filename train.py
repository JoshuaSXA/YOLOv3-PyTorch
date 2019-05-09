from model.yolo import *
from utils.data import *


def train_net(net, epochs=10, batch_size=4,lr=0.1, val_percent=0.05, save_cp=True):

    # dataloader
    dataloader = MyDataLoader("data/img/", "data/label")
    train_loader = dataloader.get_train_dataloader(batch_size=4, shuffle=True, num_works=0)
    val_loader = dataloader.get_val_dataloader(batch_size=4, num_works=0)

    # train
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        # net.train()
        for (img, label) in train_loader:
            print(label[0].size(), label[1].size(), label[2].size(), label[3].size())
            break
        break


if __name__ == '__main__':
    net = yolo_layer(input_channels=3, anchor_num=3, class_num=2)
    train_net(net)