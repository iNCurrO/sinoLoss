import numpy as np
import torch
from typing import Tuple

import customlibs.predefined_loss
from forwardprojector.FP import FP

_loss_dict = dict()


def implemented_loss_list(fn: object) -> object:
    assert callable(fn)
    _loss_dict[fn.__name__] = fn
    return fn


class total_Loss:
    def __init__(
            self,
            device,
            network,
            config,
            loss_list: Tuple[str] = tuple(["MSE"]),
            loss_weight: Tuple[float] = tuple([1.]),
    ):
        self._device = device
        self._network = network
        self._config = config
        assert self.is_available_losslist(loss_list), \
            "Implemention error: Not proper loss: {}(type:{}), where must be in {}".format(loss_list, type(loss_list), list(_loss_dict.keys()))

        self._loss_list = loss_list
        if "sinoloss_MSE" in loss_list or "sinoloss_MAE" in loss_list:
            # Generate Amatrix
            print(f"Amatrix initialization...")
            self._Amatrix = FP(config)
            print(f"Amatrix initialization finished!")
        else:
            self._Amatrix = None

        if "VGG" in loss_list:
            print(f"loading VGG loss")
            self._VGG = customlibs.predefined_loss.VGGloss().cuda()
        else:
            self._VGG = None

        if "observer" in loss_list:
            print(f"loading observer loss")
            self._observer = customlibs.predefined_loss.observerloss(config).cuda()
        else:
            self._observer = None

        self._prefunc_loss = {"Amatrix": self._Amatrix, "VGG": self._VGG, "observer": self._observer}

        if 1-sum(loss_weight) < 1e-4:
            self._loss_weight = loss_weight
        else:
            self._loss_weight = [loss_weight_element/sum(loss_weight) for loss_weight_element in loss_weight]

    @staticmethod
    def is_available_losslist(loss_list):
        return all(map(lambda x: x in list(_loss_dict.keys()), loss_list))

    def run_denoiser(self, input_img):
        return self._network(input_img)

    def run_Amatrix(self, input_img):
        return self._Amatrix(input_img)

    def accumulate_gradients(self, input_img, target_img, targetsino=None):
        logs = ""
        for idx, loss in enumerate(self._loss_list):
            with torch.autograd.profiler.record_function(loss+"_forward"):
                denoised_img = self._network(input_img.requires_grad_(True))
                temp_loss = _loss_dict[loss](
                    denoised_img=denoised_img,
                    target_img=target_img,
                    targetsino=targetsino,
                    prefunc=self._prefunc_loss,
                )
            with torch.autograd.profiler.record_function(loss+"_backward"):
                temp_loss.mul(self._loss_weight[idx]).backward()
            logs += f'Loss of {idx}.{loss}: {temp_loss} '
        return logs


@implemented_loss_list
def observer(denoised_img, target_img, targetsino=None, prefunc=None):
    return prefunc["observer"](denoised_img, target_img)


@implemented_loss_list
def VGG(denoised_img, target_img, targetsino=None, prefunc=None):
    return prefunc["VGG"](denoised_img, target_img)


@implemented_loss_list
def MSE(denoised_img, target_img,  targetsino=None, prefunc=None):
    MSEloss = torch.nn.MSELoss()
    return MSEloss(denoised_img, target_img)


@implemented_loss_list
def MAE(denoised_img, target_img, targetsino=None, prefunc=None):
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_img, target_img)


@implemented_loss_list
def sinoloss_MSE(denoised_img, target_img, targetsino=None, prefunc=None):
    denoised_sino = prefunc["Amatrix"](denoised_img)
    # sino = prefunc["Amatrix"](target_img)
    MSEloss = torch.nn.MSELoss()

    return MSEloss(denoised_sino, targetsino)


@implemented_loss_list
def sinoloss_MAE(denoised_img, target_img, targetsino=None, prefunc=None):
    denoised_sino = prefunc["Amatrix"](denoised_img)
    # sino = prefunc["Amatrix"](target_img)
    MAEloss = torch.nn.L1Loss()
    return MAEloss(denoised_sino, targetsino)
