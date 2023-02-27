import os

import torch
import argparse
from config import get_config
from model import unet
from customlibs.chores import *
from customlibs.dataset import set_dataset
from model.training_loop import training_loop
from forwardprojector.GenerateAmatrix import GenAmtrx
device = torch.device('cuda')


model_init = {
    'UNET': lambda config, img_channel: unet.Unet(n_channels=img_channel),
}


def main():
    print("Parse configurations")
    config = get_config()

    # initialize dataset
    print(f"Data initialization: {config.dataname}")
    dataloader, valdataloader, num_channels = set_dataset(config)

    # Generate Amatrix
    print(f"Amatrix initialization...")
    Amatrix = GenAmtrx(config)

    # Check Resume?
    if config.resume:
        print(f"Resume from: {config.resume}")
        __savedir__ = set_dir(config) + f"_resume{config.resume}"
        network, optimizer = resume_network(config.resume)
        # TODO make resume (Including load parameters)
        pass
    else:
        # Make dir
        __savedir__ = set_dir(config)
        print(f"logs will be archived at the {__savedir__}")

        # initialize model
        print(f"Network initialization: {config.model}")
        network = model_init[config.model.upper()](config, num_channels).cuda()
        save_parameters(network.hyperparams) # TODO make save_parameters

        # Set option
        optimizer = set_optimizer(config, network)

    training_loop(
        log_dir=__savedir__,
        training_epoch=config.trainingepoch,
        checkpoint_intvl=config.save_intlvl,
        loss_list=config.losses,
        loss_weights=config.weights,
        training_set=dataloader,
        validation_set=valdataloader,
        network=network,
        optimizer=optimizer,
        Amatrix=Amatrix,
        config=config
    )


if __name__ == "__main__":
    main()
