import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """
    Pyramid pooling modules, takes bin_sizes to make parallel pooling layer
    """
    def __init__(self, opt, in_dim, out_dim=1024):
        super(PyramidPoolingModule, self).__init__()
        self.bin_sizes = opt.bin_sizes
        self.w = opt.w
        self.h = opt.h
        self.num_label = opt.num_label

        self.modules_list = nn.ModuleList()

        for size in self.bin_sizes:
            self.modules_list.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(out_dim, out_dim)),
                nn.Conv2d(in_dim, out_dim, kernel_size=size, bias=False),
                nn.BatchNorm2d(out_dim, momentum=0.95),
                nn.ReLU(inplace=True)
            ))
        self.init_weights()

    def forward(self, in_feats):

        [h, w] = in_feats.shape[2:]
        out_feats = []
        for module in self.modules_list:
            conv_feats = module(in_feats),
            out_feats.append(F.upsample(input=conv_feats, size=(h, w), mode='bilinear'))

        return torch.cat(out_feats, 1)

    def init_weights(self):
        """
        Initialize layer weights.
        """
        for module in self.modules_list:
            for layer in module:
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.zero_()
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()

