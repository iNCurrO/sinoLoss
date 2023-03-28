from config import get_config
from model import unet
from customlibs.chores import *
from customlibs.dataset import set_dataset
from model.training_loop import training_loop
if not os.name == 'nt':
    import vessl
    device = torch.device('cuda')
    print("Initialize Vessl")
    vessl.configure(
            organization_name="yonsei-medisys",
            project_name="SinoLoss"
        )
    print()
model_init = {
    'UNET': lambda config, img_channel: unet.Unet(n_channels=img_channel),
}


def main():
    # Parse configuration
    config = get_config()

    # initialize dataset
    print(f"Data initialization: {config.dataname}\n")
    dataloader, valdataloader, num_channels = set_dataset(config)

    # Check Resume?
    if config.resume:
        print(f"Resume from: {config.resume}\n")
        __savedir__ = set_dir(config) + f"_resume{config.resume}"
        network, optimizer = resume_network(config.resume)
        # TODO make resume (Including load parameters)
        pass
    else:
        # Make dir
        __savedir__ = set_dir(config)
        print(f"logs will be archived at the {__savedir__}\n")

        # initialize model
        print(f"Network initialization: {config.model}\n")
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
        config=config
    )

    print(f"Train Done!")


if __name__ == "__main__":
    main()
