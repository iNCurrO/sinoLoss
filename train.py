from config import get_config
from model import unet
from customlibs.chores import *
from customlibs.dataset import set_dataset
from model.training_loop import training_loop

if not os.name == 'nt':
    import vessl
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

    # Initiialize model
    print(f'Network initialization: {config.mode}\n')
    network = model_init[config.model.upper()](config, num_channels)

    # initialize optimzier
    optimizer = set_optimizer(config, network)

    # Check Resume?
    if config.resume:
        print(f"Resume from: {config.resume}\n")
        __savedir__ = set_dir(config) + f"_resume{config.resume}"
        print(f"New logs will be archived at the {__savedir__}\n")
        resume_network(config.resume, network, optimizer, config)
    else:
        # Make dir
        __savedir__ = set_dir(config)
        print(f"logs will be archived at the {__savedir__}\n")

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
