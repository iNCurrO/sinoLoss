from .network_block import *
import torch.utils.checkpoint as checkpoint
import torch.nn as nn


class redcnn(nn.Module):
    def __init__(self, n_channels: int = 1, act_func: str = "GeLU", basechannel: int = 96):
        super(redcnn, self).__init__()
        self._n_channels = n_channels
        self._base_channel = basechannel
        self.down1 = nn.Conv2d(in_channels= n_channels, out_channels= basechannel, kernel_size=5, stride=1,padding=0)
        self.down2 = nn.Conv2d(in_channels= basechannel, out_channels= basechannel, kernel_size=5, stride=1,padding=0)
        self.down3 = nn.Conv2d(in_channels= basechannel, out_channels= basechannel, kernel_size=5, stride=1,padding=0)
        self.down4 = nn.Conv2d(in_channels= basechannel, out_channels= basechannel, kernel_size=5, stride=1,padding=0)
        self.down5 = nn.Conv2d(in_channels= basechannel, out_channels= basechannel, kernel_size=5, stride=1,padding=0)
        
        self.up1 = nn.ConvTranspose2d(in_channels=basechannel, out_channels=basechannel, kernel_size=5, stride=1, padding=0)
        self.up2 = nn.ConvTranspose2d(in_channels=basechannel, out_channels=basechannel, kernel_size=5, stride=1, padding=0)
        self.up3 = nn.ConvTranspose2d(in_channels=basechannel, out_channels=basechannel, kernel_size=5, stride=1, padding=0)
        self.up4 = nn.ConvTranspose2d(in_channels=basechannel, out_channels=basechannel, kernel_size=5, stride=1, padding=0)
        self.up5 = nn.ConvTranspose2d(in_channels=basechannel, out_channels=n_channels, kernel_size=5, stride=1, padding=0)

        if act_func == "ReLU":
            self._act_func = nn.ReLU(inplace=True)
        elif act_func == "LReLU":
            self._act_func = nn.LeakyReLU(inplace=True)
        elif act_func == "GeLU":
            self._act_func = nn.GELU()
        else:
            print(f"Unsupported activation module: {act_func}")
            raise NotImplementedError
        

    def base_channel(self):
        return self._base_channel

    def hyperparams(self):
        return {'n_channels': self._n_channels, 'basechannel': self._base_channel, 'act_func': self._act_func}

    def forward(self, x):
        # =============================================
        # Encoder part
        residual_1 = x
        x= self._act_func(self.down1(x))
        x= self._act_func(self.down2(x))
        residual_2 = x
        x= self._act_func(self.down3(x))
        x= self._act_func(self.down4(x))
        residual_3 = x
        x= self._act_func(self.down5(x))
        
        # ==============================================
        # Decoder part
        x = self.up1(x)
        x += residual_3
        x = self.up2(self._act_func(x))
        x = self.up3(self._act_func(x))
        x += residual_2
        x = self.up4(self._act_func(x))
        x = self.up5(self._act_func(x))
        x += residual_1
        out = self._act_func(x)
        return out

