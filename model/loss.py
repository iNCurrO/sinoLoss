import numpy as np
import torch


class Loss:
    def accumulate_gradients(self, epoch: int, input_img, target_img, input_sino, target_sino):
        raise NotImplementedError()


class MSE_loss(Loss):
    def __init__(
            self, device, network
    ):
        super().__init__()
        self._device = device
        self._network = network

    def accumulate_gradients(self, epoch: int, input_img, target_img, input_sino, target_sino):
        pass
