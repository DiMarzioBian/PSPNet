import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.metrics import LabelSmoothingLoss


class SpatialPyramidPool2D(nn.Module):
    def __init__(self):
        super(SpatialPyramidPool2D, self).__init__()
        self.local_avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # local avg pooling, gives 512@2*2 feature
        features_local = self.local_avg_pool(x)
        # global avg pooling, gives 512@1*1 feature
        features_pool = self.global_avg_pool(x)
        # flatten and concatenate
        out1 = features_local.view(features_local.size()[0], -1)
        out2 = features_pool.view(features_pool.size()[0], -1)
        return torch.cat((out1, out2), 1)


class SPP_Resnet(nn.Module):
    def __init__(self, num_label, enable_spp=True):
        super(SPP_Resnet, self).__init__()

        if enable_spp:
            arch = list(models.resnet50(pretrained=True).children())
            self.model = nn.Sequential(
                nn.Sequential(*arch[:-3]),
                arch[-3:-2][0][0],
                nn.Sequential(*list(arch[-3:-2][0][1].children())[:-1]),
                SpatialPyramidPool2D(),
                nn.Linear(2560, num_label, bias=True)
            )
        else:
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, num_label, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class PSPNet(nn.Module):
    def __init__(self, opt):
        super(PSPNet, self).__init__()
        self.enable_spp = opt.enable_spp
        self.num_label = opt.num_label

        self.spp_resnet = SPP_Resnet(self.num_label, enable_spp=self.enable_spp)

        # self.sincNet1 = nn.Sequential(
        #     nn.Conv2d(out_channels=160, kernel_size=251),
        #     nn.BatchNorm1d(160),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool1d(1024))
        # self.sincNet2 = nn.Sequential(
        #     nn.Conv2d(out_channels=160, kernel_size=501),
        #     nn.BatchNorm1d(160),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool1d(1024))
        # self.sincNet3 = nn.Sequential(
        #     nn.Conv2d(out_channels=160, kernel_size=1001),
        #     nn.BatchNorm1d(160),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool1d(1024))

        self.calc_loss = LabelSmoothingLoss(opt.smooth_label, opt.num_label)

    def forward(self, x):
        """ Feature extraction """
        # x = self.layerNorm(x)

        # feat1 = self.sincNet1(x)
        # feat2 = self.sincNet2(x)
        # feat3 = self.sincNet3(x)
        #
        # x = torch.cat((feat1.unsqueeze_(dim=1),
        #                feat2.unsqueeze_(dim=1),
        #                feat3.unsqueeze_(dim=1)), dim=1)
        x = self.spp_resnet(x)
        return x


