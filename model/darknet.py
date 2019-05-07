from model.common import *

class darknet(nn.Module):
    def __init__(self, input_channels):
        super(darknet, self).__init__()
        self.dark_conv = dark_conv(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.res_block1 = res_block(in_channels=32, out_channels=64, num_blocks=1)
        self.res_block2 = res_block(in_channels=64, out_channels=128, num_blocks=2)
        self.res_block3 = res_block(in_channels=128, out_channels=256, num_blocks=8)
        self.res_block4 = res_block(in_channels=256, out_channels=512, num_blocks=8)
        self.res_block5 = res_block(in_channels=512, out_channels=1024, num_blocks=4)

    def forward(self, *x):
        out = self.dark_conv(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out1 = self.res_block3(out)
        out2 = self.res_block4(out1)
        out3 = self.res_block5(out2)
        return out1, out2, out3


class tail_block(nn.Module):
    def __init__(self, in_channels, filter_num, out_channels):
        super(tail_block, self).__init__()
        self.multi_conv = nn.Sequential(
            dark_conv(in_channels=in_channels, out_channels=filter_num, kernel_size=1, padding=0),
            dark_conv(in_channels=filter_num, out_channels=filter_num * 2, kernel_size=3, padding=1),
            dark_conv(in_channels=filter_num * 2, out_channels=filter_num, kernel_size=1, padding=0),
            dark_conv(in_channels=filter_num, out_channels=filter_num * 2, kernel_size=3, padding=1),
            dark_conv(in_channels=filter_num * 2, out_channels=filter_num, kernel_size=1, padding=0)
        )
        self.out_conv = nn.Sequential(
            dark_conv(in_channels=filter_num, out_channels=filter_num * 2, kernel_size=3, padding=1),
            nn.Conv2d(filter_num * 2, out_channels, kernel_size=1)
        )

    def forward(self, *x):
        out1 = self.multi_conv(x)
        out2 = self.out_conv(out1)
        return out1, out2


class yolo3(nn.Module):
    def __init__(self, input_channels, anchor_num, class_num):
        super(yolo3, self).__init__()
        self.darknet = darknet(input_channels)
        self.tail_block1 = tail_block(1024, 512, anchor_num * (class_num + 5))
        self.tail_block2 = tail_block(512, 256, anchor_num * (class_num + 5))
        self.tail_block3 = tail_block(256, 128, anchor_num * (class_num + 5))
        self.up_block1 = nn.Sequential(
            dark_conv(in_channels=1024, out_channels=256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2)
        )
        self.up_block2 = nn.Sequential(
            dark_conv(in_channels=256, out_channels=128, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, *x):
        dark_out1, dark_out2, dark_out3 = self.darknet(x)
        out, out1 = self.tail_block1(dark_out3)
        out = torch.cat([self.up_block1(out), dark_out2], dim=1)
        out, out2 = self.tail_block2(out)
        out = torch.cat([self.up_block2(out), dark_out1], dim=1)
        _, out3 = self.tail_block3(out)
        return out1, out2, out3
