import torch
import torch.nn as nn


_act_func_dict = dict()


def register_act_func(fn: object) -> object:
    assert callable(fn)
    _act_func_dict[fn.__name__] = fn
    return fn


class DoubleConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            act_func: str = 'ReLU'
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            _act_func_dict[act_func],
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _act_func_dict[act_func],
        )

    def forward(self, x):
        return self.layers(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels
            )
        )

    def forward(self, x):
        return self.layers(x)


class UpSampling(nn.Module):
    @staticmethod
    def is_available_interp(interp: str):
        return interp in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']

    def __init__(self, in_channels: int, out_channels: int, interp: str='bilinear'):
        super().__init__()
        if self.is_available_interp(interp):
            self.upSampling = nn.Upsample(scale_factor=2, mode=interp, align_corners=True)
            self.convBlock = DoubleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=int(in_channels/2)
            )
        else:
            self.upSampling = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=int(in_channels/2),
                kernel_size=(2, 2),
                stride=(2, 2)
            )
            self.convBlock = DoubleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
            )

    def forward(self, x, skippeddata):
        upsampled_x = self.upSampling(x)
        x = torch.cat([skippeddata, upsampled_x])
        return self.convBlock(x)


class LastConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convBlock = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        return self.conv(x)


@register_act_func
def ReLU():
    return nn.ReLU(inplace=True)


@register_act_func
def ELU():
    return nn.ELU(alpha=1.0, inplace=True)


@register_act_func
def GELU():
    return nn.GELU()


@register_act_func
def LeakyReLU():
    return nn.LeakyReLU()
