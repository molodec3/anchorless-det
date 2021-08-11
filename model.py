import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class CenterNet(nn.Module):
    def __init__(self, n_classes, resnet=18, pretrained=False):
        super(CenterNet, self).__init__()

        self.n_classes = n_classes

        if resnet == 18:
            backbone = models.resnet18(pretrained=pretrained)
        elif resnet == 34:
            backbone = models.resnet34(pretrained=pretrained)
        elif resnet == 50:
            backbone = models.resnet50(pretrained=pretrained)
        elif resnet == 101:
            backbone = models.resnet101(pretrained=pretrained)
        elif resnet == 152:
            backbone = models.resnet152(pretrained=pretrained)
        else:
            raise AttributeError('wrong resnet backbone')

        self.backbone_l1 = nn.Sequential(*list(backbone.children())[:5])
        self.backbone_l2 = list(backbone.children())[5]
        self.backbone_l3 = list(backbone.children())[6]
        self.backbone_l4 = list(backbone.children())[7]

        if resnet <= 34:
            self.deconv_l1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.deconv_l2 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
            self.deconv_l3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
            self.pred = nn.Conv2d(128, self.n_classes + 4, kernel_size=3, padding=1)
        else:
            self.deconv_l1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
            self.deconv_l2 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
            self.deconv_l3 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
            self.pred = nn.Conv2d(512, self.n_classes + 4, kernel_size=3, padding=1)

    def forward(self, inp):
        inp1 = self.backbone_l1(inp)
        inp2 = self.backbone_l2(inp1)
        inp3 = self.backbone_l3(inp2)
        inp4 = self.backbone_l4(inp3)

        out = self.deconv_l1(inp4)
        out = self.deconv_l2(torch.cat([inp3, out], dim=1))
        out = self.deconv_l3(torch.cat([inp2, out], dim=1))
        out = self.pred(torch.cat([inp1, out], dim=1))

        # sigmoid to have values in [0, 1]
        return torch.sigmoid(out[:, :self.n_classes]), \
            out[:, self.n_classes:-2], \
            out[:, -2:]
