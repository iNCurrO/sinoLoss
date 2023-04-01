from .network_block import *
import torch.utils.checkpoint as checkpoint
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, n_channels: int, interp: str = None, act_func: str = "ReLU"):
        super(Unet, self).__init__()
        self._n_channels = n_channels
        self._interp = interp
        self._act_func = act_func

        self.inc = (DoubleConvBlock(n_channels, 64, act_func=self._act_func))
        self.down1 = (DownSampling(64, 128, act_func=self._act_func))
        self.down2 = (DownSampling(128, 256, act_func=self._act_func))
        self.down3 = (DownSampling(256, 512, act_func=self._act_func))
        factor = 2 if UpSampling.is_available_interp(interp=self._interp) else 1
        self.down4 = (DownSampling(512, 1024 // factor, act_func=self._act_func))
        self.up1 = (UpSampling(1024, 512 // factor, self._interp, act_func=self._act_func))
        self.up2 = (UpSampling(512, 256 // factor, self._interp, act_func=self._act_func))
        self.up3 = (UpSampling(256, 128 // factor, self._interp, act_func=self._act_func))
        self.up4 = (UpSampling(128, 64, self._interp, act_func=self._act_func))
        self.outc = (LastConv(64, self._n_channels))

    def hyperparams(self):
        return {'n_channels': self._n_channels, 'interp': self._interp, 'act_func': self._act_func}

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = checkpoint.checkpoint(self.inc)
    #     self.down1 = checkpoint.checkpoint(self.down1)
    #     self.down2 = checkpoint.checkpoint(self.down2)
    #     self.down3 = checkpoint.checkpoint(self.down3)
    #     self.down4 = checkpoint.checkpoint(self.down4)
    #     self.up1 = checkpoint.checkpoint(self.up1)
    #     self.up2 = checkpoint.checkpoint(self.up2)
    #     self.up3 = checkpoint.checkpoint(self.up3)
    #     self.up4 = checkpoint.checkpoint(self.up4)
    #     self.outc = checkpoint.checkpoint(self.outc)

