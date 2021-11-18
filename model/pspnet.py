import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import resnet34
from model.layers import PyramidPoolingModule
from model.metrics import LabelSmoothingLoss


class Backbone(nn.Module):
    def __init__(self, backbone, num_label):
        super(Backbone,  self).__init__()
        if backbone == 'resnet50':
            self.model = resnet34(pretrained=True, replace_stride_with_dilation=[0, 2, 4])

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_label):
        super(Classifier,  self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_label, kernel_size=1),
        )
        self.init_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self):
        """
        Initialize layer weights.
        """
        for layer in self.model:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self,
                 opt,
                 psp_size=2048,
                 deep_features_size=1024,
                 ):
        super(PSPNet, self).__init__()
        self.enable_spp = opt.enable_spp
        self.num_label = opt.num_label
        self.backbone = opt.backbone

        # Overrider Resnet official code, add dilation at BasicBlock 3 and 4 according to paper
        self.backbone = Backbone(self.backbone, self.num_label)

        self.pyramid_pooling = PyramidPoolingModule(opt, in_dim=2048, out_dim=512)
        self.pyramid_pooling.init_weights()

        self.classifier = Classifier(self.num_label)

    def forward(self, x, x2):
        x = self.backbone(x)
        x = self.pyramid_pooling(x)
        x = self.classifier(x)
        return x


