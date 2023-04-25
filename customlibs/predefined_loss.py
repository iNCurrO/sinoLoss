import os.path

import torch
import torchvision
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import conv2d as Fconv2d
from torch.nn.functional import leaky_relu as FLReLU
from torch.nn.functional import max_pool2d as Fpool
import h5py


class observerloss(torch.nn.Module):
    def __init__(self, config):
        super(observerloss, self).__init__()
        try:
            with h5py.File(os.path.join(config.datadir, 'observerloss.h5'), 'r') as w:
                self._conv1_w = torch.from_numpy(w['model_weights']['observer_core']['conv1']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv1_b = torch.from_numpy(w['model_weights']['observer_core']['conv1']['bias:0'][:]).cuda()
                self._conv2_w = torch.from_numpy(w['model_weights']['observer_core']['conv2']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv2_b = torch.from_numpy(w['model_weights']['observer_core']['conv2']['bias:0'][:]).cuda()
                self._conv3_w = torch.from_numpy(w['model_weights']['observer_core']['conv3']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv3_b = torch.from_numpy(w['model_weights']['observer_core']['conv3']['bias:0'][:]).cuda()
                self._conv4_w = torch.from_numpy(w['model_weights']['observer_core']['conv4']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv4_b = torch.from_numpy(w['model_weights']['observer_core']['conv4']['bias:0'][:]).cuda()
                self._conv5_w = torch.from_numpy(w['model_weights']['observer_core']['conv5']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv5_b = torch.from_numpy(w['model_weights']['observer_core']['conv5']['bias:0'][:]).cuda()
                self._conv6_w = torch.from_numpy(w['model_weights']['observer_core']['conv6']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv6_b = torch.from_numpy(w['model_weights']['observer_core']['conv6']['bias:0'][:]).cuda()
                self._conv7_w = torch.from_numpy(w['model_weights']['observer_core']['conv7']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv7_b = torch.from_numpy(w['model_weights']['observer_core']['conv7']['bias:0'][:]).cuda()
                self._conv8_w = torch.from_numpy(w['model_weights']['observer_core']['conv8']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv8_b = torch.from_numpy(w['model_weights']['observer_core']['conv8']['bias:0'][:]).cuda()
                self._conv9_w = torch.from_numpy(w['model_weights']['observer_core']['conv9']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv9_b = torch.from_numpy(w['model_weights']['observer_core']['conv9']['bias:0'][:]).cuda()
                self._conv10_w = torch.from_numpy(w['model_weights']['observer_core']['conv10']['kernel:0'][:]).permute(3, 2, 0, 1).cuda()
                self._conv10_b = torch.from_numpy(w['model_weights']['observer_core']['conv10']['bias:0'][:]).cuda()

        except FileNotFoundError:
            print(f"observer loss weights are not founded: {os.path.join(config.datadir, 'observerloss.h5')}\n")

    def forward(self, input_img, target_img):
        loss = 0.0
        x = input_img
        x = Fconv2d(x, self._conv1_w, bias=self._conv1_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv2_w, bias=self._conv2_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv3_w, bias=self._conv3_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv4_w, bias=self._conv4_b)
        x = FLReLU(x)
        x = Fpool(x, (2, 2))
        y = target_img
        y = Fconv2d(y, self._conv1_w, bias=self._conv1_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv2_w, bias=self._conv2_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv3_w, bias=self._conv3_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv4_w, bias=self._conv4_b)
        y = FLReLU(y)
        y = Fpool(y, (2, 2))
        loss += mse_loss(x, y)

        x = Fconv2d(x, self._conv5_w, bias=self._conv5_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv6_w, bias=self._conv6_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv7_w, bias=self._conv7_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv8_w, bias=self._conv8_b)
        x = FLReLU(x)
        y = Fconv2d(y, self._conv5_w, bias=self._conv5_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv6_w, bias=self._conv6_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv7_w, bias=self._conv7_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv8_w, bias=self._conv8_b)
        y = FLReLU(y)
        loss += mse_loss(x, y)

        x = Fpool(x, (2, 2), stride=(2, 2))
        x = Fconv2d(x, self._conv9_w, bias=self._conv9_b)
        x = FLReLU(x)
        x = Fconv2d(x, self._conv10_w, bias=self._conv10_b)
        x = FLReLU(x)
        y = Fpool(y, (2, 2), stride=(2, 2))
        y = Fconv2d(y, self._conv9_w, bias=self._conv9_b)
        y = FLReLU(y)
        y = Fconv2d(y, self._conv10_w, bias=self._conv10_b)
        y = FLReLU(y)
        loss += mse_loss(x, y)

        return loss


class VGGloss(torch.nn.Module):
    def __init__(self):
        super(VGGloss, self).__init__()
        blocks = [torchvision.models.vgg16(pretrained=True).features[:4].eval(),
                  torchvision.models.vgg16(pretrained=True).features[4:9].eval(),
                  torchvision.models.vgg16(pretrained=True).features[9:16].eval(),
                  torchvision.models.vgg16(pretrained=True).features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.required_grad = False
        self._blocks = torch.nn.ModuleList(blocks)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img, feature_layers=None, style_layers=None):
        if feature_layers is None:
            feature_layers = [0, 1, 2, 3]
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std
        x = input_img
        y = target_img
        loss = 0.0
        for i, block in enumerate(self._blocks):
            x = block(x)
            y = block(y)
            if feature_layers:
                if i in feature_layers:
                    b, c, h, w = x.shape
                    loss += l1_loss(x, y)
            if style_layers:
                if i in style_layers:
                    b, c, h, w = x.shape
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1) / (c * h * w)
                    gram_y = act_y @ act_y.permute(0, 2, 1) / (c * h * w)
                    loss += l1_loss(gram_x, gram_y)
        return loss
