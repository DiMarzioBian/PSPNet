import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import resnet34
from model.layers import PyramidPoolingModule
from model.metrics import LabelSmoothingLoss


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone,  self).__init__()
        if backbone == 'resnet50':
            self.model = resnet34(pretrained=True, replace_stride_with_dilation=[0, 2, 4])

    def forward(self, x):
        x, x_auxiliary = self.model(x)
        return x, x_auxiliary


class Classifier(nn.Module):
    """
    Classifier for pyramid pooling features
    """
    def __init__(self, in_dim, mid_dim, num_label):
        super(Classifier,  self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_dim, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_dim, num_label, kernel_size=(1, 1)),
        )
        self.init_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class Classifier_auxiliary(nn.Module):
    """
    Classifier for auxiliary features
    """
    def __init__(self, in_dim, num_label):
        super(Classifier_auxiliary,  self).__init__()
        self.conv = nn.Conv2d(in_dim, num_label, kernel_size=(1, 1), padding=1)
        self.init_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self):
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self,
                 opt,
                 ):
        super(PSPNet, self).__init__()
        self.enable_spp = opt.enable_spp
        self.num_label = opt.num_label
        self.pooling_dim = opt.pooling_dim
        self.backbone = opt.backbone

        # Overrider Resnet official code, add dilation at BasicBlock 3 and 4 according to paper
        self.backbone = Backbone(self.backbone)

        self.pyramid_pooling = PyramidPoolingModule(opt, in_dim=opt.out_dim_resnet, out_dim=self.pooling_dim)
        self.pyramid_pooling.init_weights()

        in_dim_classifier = opt.out_dim_resnet + len(opt.bin_sizes) * self.pooling_dim
        self.classifier = Classifier(in_dim=in_dim_classifier, mid_dim=512, num_label=self.num_label)
        if opt.backbone == 'resnet50':
            in_dim_auxiliary = 256
        self.classifier_auxiliary = Classifier_auxiliary(in_dim=in_dim_classifier, num_label=self.num_label)

    def forward(self, img):
        x, x_auxiliary = self.backbone(img)
        x = self.pyramid_pooling(x)
        x = self.classifier(x)
        x_auxiliary = self.classifier_auxiliary(x_auxiliary)

        x = nn.functional.interpolate(x, img.shape[2:], mode='bilinear', align_corners=False)
        x_auxiliary = nn.functional.interpolate(x_auxiliary, img.shape[2:], mode='bilinear', align_corners=False)
        return x, x_auxiliary




