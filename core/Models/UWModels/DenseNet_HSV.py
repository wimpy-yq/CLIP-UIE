import torch
import torch.nn as nn
import torch.nn.functional as F
from SAM.segment_anything.utils.transforms import ResizeLongestSide
from SAM.segment_anything import sam_model_registry, SamPredictor
from typing import Type, Optional, Tuple, Any
from .common import LayerNorm2d
from core.Models.UWModels.SAMUnet1 import ColorNet




class SAMUNet(nn.Module):
    def __init__(self, sam_model):
        super(SAMUNet, self).__init__()
        self.sam_model = sam_model
        self.image_encoder = sam_model.image_encoder
        # 冻结SAM模型参数
        for param in self.sam_model.parameters():
            param.requires_grad = False

    def sam_enhance_from_torch(self, image):
        if self.sam_model is not None:
            local_rank = image.device  # 从输入图像张量获取设备信息
            sam_transform = ResizeLongestSide(self.image_encoder.img_size)
            resampled_image = sam_transform.apply_image_torch(image).cuda(local_rank)
            resampled_image = self.sam_model.preprocess(resampled_image)

            assert resampled_image.shape == (resampled_image.shape[0], 3, self.image_encoder.img_size,
                                             self.image_encoder.img_size), '输入图像应调整为3*1024*1024'

            with torch.no_grad():
                embedding = self.image_encoder(resampled_image)

            return embedding

    def forward(self, image):
        return self.sam_enhance_from_torch(image)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)




class DenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DenseUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        # self.up3 = Up_0(256, 256,128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # print(x3.shape)256 128 128
        # print(x_sam.shape)256 64 64
        x4 = self.down3(x3)
        # print(x4.shape)512 64 64
        # print(x_sam.shape)256 64 64
        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        # x = self.up2(x, x_sam, x3)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class DenseHSV(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(DenseHSV, self).__init__()
        self.dense_unet = DenseUNet(in_channels, out_channels)
        self.color_net = ColorNet()

    def forward(self, image):
        output = self.dense_unet(image)  # 将特征传递给 DenseUNet
        output = self.color_net(image,output)
        return output


# 示例用法
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# sam_model = load_sam_model(sam_checkpoint='path_to_checkpoint', device=device, model_type='your_model_type')
# model = CombinedModel(sam_model, in_channels=3, out_channels=3).to(device)
# input_tensor = torch.randn((1, 3, 512, 512)).to(device)  # 示例输入图像
# output = model(input_tensor)
# print(output.shape)  # 应输出: torch.Size([1, 3, 512, 512])
