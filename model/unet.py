from .network_block import *
import torch.utils.checkpoint as checkpoint
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, n_channels: int, interp: str = None, act_func: str = "ReLU", basechannel=64):
        super(Unet, self).__init__()
        self._n_channels = n_channels
        self._interp = interp
        self._act_func = act_func
        self._base_channel = basechannel

        self.inc = (DoubleConvBlock(n_channels, self._base_channel, act_func=self._act_func))
        self.down1 = (DownSampling(self._base_channel, 2*self._base_channel, act_func=self._act_func))
        self.down2 = (DownSampling(2*self._base_channel, (2**2)*self._base_channel, act_func=self._act_func))
        self.down3 = (DownSampling((2**2)*self._base_channel, (2**3)*self._base_channel, act_func=self._act_func))
        factor = 2 if UpSampling.is_available_interp(interp=self._interp) else 1
        self.down4 = (DownSampling((2**3)*self._base_channel, (2**4)*self._base_channel // factor, act_func=self._act_func))
        self.up1 = (UpSampling((2**4)*self._base_channel, (2**3)*self._base_channel // factor, self._interp, act_func=self._act_func))
        self.up2 = (UpSampling((2**3)*self._base_channel, (2**2)*self._base_channel // factor, self._interp, act_func=self._act_func))
        self.up3 = (UpSampling((2**2)*self._base_channel, 2*self._base_channel // factor, self._interp, act_func=self._act_func))
        self.up4 = (UpSampling(2*self._base_channel, self._base_channel, self._interp, act_func=self._act_func))
        self.outc = (LastConv(self._base_channel, self._n_channels))

    def base_channel(self):
        return self._base_channel

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

