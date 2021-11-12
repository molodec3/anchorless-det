import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class CenterNet(nn.Module):
    def __init__(
        self, n_classes, 
        resnet=18, pretrained=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(CenterNet, self).__init__()

        self.n_classes = n_classes
        self.device = device

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
        # returns heatmap, offsets, sizes
        return torch.sigmoid(out[:, :self.n_classes]), \
            out[:, self.n_classes:-2], \
            out[:, -2:]
    
    def predict(
        self, inp, 
        thr=0.5, num_peaks=100, 
        downsampling_ratio=4,
    ):
        hm, offset, size = self.forward(inp)
        
        b, c, h, w = hm.shape
        
        kernel = 3
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            hm, (kernel, kernel), stride=1, padding=pad
        )
        keep = (hmax == hm).float()
        hm = keep * hm
        
        peaks, p_idxs = torch.topk(hm.flatten(-3), num_peaks)
        stacked = torch.arange(b).reshape(-1, 1)
        stacked = stacked.repeat(1, num_peaks).reshape(-1, 1)
        
        indices = tuple(
            torch.cat(
                [stacked, p_idxs.reshape(-1, 1)], dim=-1
            )[peaks.reshape(-1, ) > thr].T
        )
        
        unraveled = np.unravel_index(indices[1].cpu().detach().numpy(), (c, h, w))
        classes, ys, xs = unraveled
        
        # add offset and find size of x and y
        ys_idxs = (
            indices[0], torch.zeros(indices[0].shape, dtype=int).to(self.device), 
            torch.tensor(ys).to(self.device), torch.tensor(xs).to(self.device)
        )
        xs_idxs = (
            indices[0], torch.ones(indices[0].shape, dtype=int).to(self.device), 
            torch.tensor(ys).to(self.device), torch.tensor(xs).to(self.device)
        )
        
        upd_ys = torch.tensor(ys).to(self.device) + offset[ys_idxs]
        upd_xs = torch.tensor(xs).to(self.device) + offset[xs_idxs]
                
        lower_ys = upd_ys - size[ys_idxs] / 2
        upper_ys = upd_ys + size[ys_idxs] / 2
        lower_xs = upd_xs - size[xs_idxs] / 2
        upper_xs = upd_xs + size[xs_idxs] / 2
        
        # returns tensor with shape [bs * n_peaks, 6]
        # containing [batch_num, class_num, x_min, y_min, x_max, y_max]
        print(indices[0].shape, torch.tensor(classes).to(self.device).shape, lower_xs.shape)
        return torch.cat([
            indices[0].reshape(-1, 1), 
            torch.tensor(classes).to(self.device).reshape(-1, 1), 
            4 * lower_xs.reshape(-1, 1), 
            4 * lower_ys.reshape(-1, 1), 
            4 * upper_xs.reshape(-1, 1), 
            4 * upper_ys.reshape(-1, 1)
        ], dim=-1)
