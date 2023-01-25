import time
from typing import Tuple

from tqdm import tqdm
from customlibs.chores import save_network, save_images
from model.loss import *


def training_loop(
        log_dir: str = "./log",  # log dir
        training_epoch: int = 50,  # The number of iteration epochs
        loss_list: Tuple[str] = tuple(["MSE"]),  # List of used loss (Type: tuple of strings)
        loss_weights: Tuple[float] = tuple([1.]),  # List of loss weight (Type: tuple of floats)
        checkpoint_intvl: int = 5,  # Interval of saving checkpoint (in epochs)
        training_set=None,  # Dataloader of training set
        validation_set=None,  # Dataloader of validation set
        network=None,  # Constructed network
        optimizer=None,  # Used optimizer
        config=None,
):
    device = torch.device('cuda')

    # Print parameters
    print()
    print('Num of training images: ', len(training_set.dataset))
    print("loss_list: ", loss_list)
    print()

    # Constructing losses
    print("Constructing losses....")
    loss_func = total_Loss(device=device, network=network, config=config, loss_list=loss_list, loss_weight=loss_weights)

    # Initialize logs

    # Train
    print("Start Training...\nSaving Initial samples")
    val_noisy_img, val_target_img = next(iter(validation_set))
    val_batch_size = val_noisy_img.shape[0]
    save_images(val_target_img, epoch=0, tag="target", savedir=log_dir, batchnum=val_batch_size)
    save_images(val_noisy_img, epoch=0, tag='noisy', savedir=log_dir, batchnum=val_batch_size)
    network.eval()
    val_denoised_img = network(val_noisy_img.cuda())
    save_images(
        val_denoised_img.cpu().detach().numpy(), epoch=0, tag="denoised", savedir=log_dir, batchnum=val_batch_size
    )
    network.train().requires_grad_(True)
    start_time = time.time()
    cur_time = time.time()
    for cur_epoch in range(training_epoch):
        # iteration for one epcoh
        with tqdm(training_set) as pbar:
            for batch_idx, samples in enumerate(pbar):
                optimizer.zero_grad()
                [noisy_img, sino, target_img] = samples
                logs = loss_func.accumulate_gradients(noisy_img.cuda(), target_img.cuda(), sino.cuda())
                pbar.set_description(
                    f'Train Epoch: {cur_epoch}/{training_epoch},' +
                    f'mean(sec/batch): {(cur_time-start_time)/cur_epoch if cur_epoch else 0}, loss:' +
                    str(logs) +
                    f'ETA: {(cur_time-start_time)/cur_epoch*(training_epoch-cur_epoch) if cur_epoch else 0}'
                )
                with torch.autograd.profiler.record_function("opt"):
                    optimizer.step()

        # Save check point and evaluate
        cur_time = time.time()
        if cur_epoch % checkpoint_intvl == 0 and cur_epoch != 0:
            network.eval()
            val_denoised_img = network(val_noisy_img.cuda())
            save_network(network=network, epoch=cur_epoch, savedir=log_dir)
            save_images(
                val_denoised_img.cpu().detach().numpy(),
                epoch=cur_epoch,
                tag="denoised",
                savedir=log_dir,
                batchnum=val_batch_size
            )
            network.train()
