from config import get_config
from model import unet, redcnn
from customlibs.chores import *
from customlibs.dataset import set_dataset
from model.training_loop import training_loop
from evaluate import evaluate_main

if not os.name == 'nt':
    import vessl
    print("Initialize Vessl")
    vessl.configure(
            organization_name="yonsei-medisys",
            project_name="sino-domain-loss",
        )
    print()
model_init = {
    'UNET': lambda config, img_channel: unet.Unet(n_channels=img_channel),
    'REDCNN': lambda config, img_channel: redcnn.redcnn(n_channels=img_channel, basechannel=config.base_channel, act_func=config.act_func)
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

    __savedir__, __dirnum__ = set_dir(config)
    # Check Resume?
    if config.resume:
        print(f"New logs will be archived at the {__savedir__}\n")
        print("Loading.... network")
        resume_network(config.resume, network, optimizer, config)
        print("loaded!")
    else:
        # Make dir
        print(f"logs will be archived at the {__savedir__}\n")

    if not os.name == 'nt':
        hp = {
            "optimizer": config.optimizer,
            "LR": config.learningrate,
            "LR_decay": config.lrdecay,
            "model_size": network.base_channel(),
            "Resume_from": config.resume
        }
        vessl.hp.update(hp)

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

    evaluate_main(resumenum=str(__dirnum__)+'-'+str(config.trainingepoch), __savedir__=__savedir__)
    print(f"Testing Done!")
    if not os.name == 'nt':
        vessl.finish()


if __name__ == "__main__":
    main()
