import os.path
import time

import torch

if not os.name == 'nt':
    import vessl
from customlibs.chores import save_network, save_images, lprint
from customlibs.metrics import *
from model.loss import *
from evaluate import evaluate
from datetime import timedelta

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
    device = torch.device(config.device)
    network = network.to(device)

    # Print parameters
    print()
    print('Num of training images: ', len(training_set.dataset))
    print("loss_list: ", loss_list)
    print()

    # Generate Amatrix
    print(f"Amatrix initialization...")
    Amatrix = FP(config)
    print(f"Amatrix initialization finished!")

    # Constructing losses
    print("Constructing losses....")
    loss_func = total_Loss(
        device=device, network=network, config=config, loss_list=loss_list, loss_weight=loss_weights, Amatrix=Amatrix
    )

    # Initialize logs
    with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
        print("Start of Files ================== \n", file=log_file)


    # Train
    val_noisy_img, val_target_sino, val_target_img = next(iter(validation_set))
    val_batch_size = val_noisy_img.shape[0]
    save_images(val_target_img, epoch=0, tag="target", savedir=log_dir, batchnum=val_batch_size)
    save_images(val_noisy_img, epoch=0, tag='noisy', savedir=log_dir, batchnum=val_batch_size)
    network.eval()
    val_denoised_img = network(val_noisy_img.to(device))
    save_images(
        val_denoised_img.cpu().detach().numpy(), epoch=0, tag="denoised", savedir=log_dir, batchnum=val_batch_size
    )
    # save_images(
    #     loss_func.run_Amatrix(val_target_img.to(device)).cpu().detach().numpy(),
    #     epoch=0, tag="target_sino", savedir=log_dir, batchnum=val_batch_size, sino=True
    # )
    # save_images(
    #     loss_func.run_Amatrix(val_noisy_img.to(device)).cpu().detach().numpy(),
    #     epoch=0, tag='noisy_sino', savedir=log_dir, batchnum=val_batch_size, sino=True
    # ) # TODO
    network.train().requires_grad_(True)

    # Main Part
    start_time = time.time()
    lprint(
        f"Entering training at {time.localtime(start_time).tm_mon}/{time.localtime(start_time).tm_mday} "
        f"{time.localtime(start_time).tm_hour}h {time.localtime(start_time).tm_min}m "
        f"{time.localtime(start_time).tm_sec}s",
        log_dir=log_dir
    )

    for cur_epoch in range(training_epoch):
        # iteration for one epcoh
        logs = ""
        for batch_idx, samples in enumerate(training_set):
            optimizer.zero_grad()
            [noisy_img, sino, target_img] = samples
            logs = loss_func.accumulate_gradients(
                noisy_img.to(device),
                target_img.to(device),
                targetsino=sino.to(device)
            )
            if batch_idx % 99 == 0:
                nettime = time.time() - start_time
                realtime_epoch = (cur_epoch + ((batch_idx+config.batchsize) / len(training_set)))
                lprint(
                    f'Train Epoch: {cur_epoch}/{training_epoch}, Batch: {batch_idx}/{len(training_set)}' +
                    f'mean(sec/epoch): '
                    f'{nettime / realtime_epoch}'
                    f', loss:' +
                    str(logs) +
                    f'ETA: {timedelta(seconds=(nettime / realtime_epoch * (training_epoch - realtime_epoch)) if not (cur_epoch==0 and batch_idx==0) else 0)}',
                    log_dir=log_dir
                )
            optimizer.step()
        vessl.log(step=cur_epoch, payload={"trainingloss_"+keys: logs[keys] for keys in logs})

        # Save check point and evaluate
        network.eval()
        with torch.no_grad():
            val_denoised_img = network(val_noisy_img.to(device))
            if cur_epoch%5 == 0:
                val_ssim, val_psnr, val_mse, val_sinomse = evaluate(network, validation_set, Amatrix)
                # Print log
                lprint(
                    f'==========================================================================\n' +
                    f'Evaluation for Epoch: {cur_epoch}/{training_epoch},' +
                    f'mean(sec/Epoch): {(time.time() - start_time) / (cur_epoch+1)}, loss:' +
                    str(logs) + '\n' +
                    f'metrics: SSIM [{val_ssim}], '
                    f'PSRN [{val_psnr}], '
                    f'MSE: [{val_mse}], '
                    f'sinoMSE: [{val_sinomse}]',
                    log_dir=log_dir
                )
                if not os.name == 'nt':
                    vessl.log(step=cur_epoch, payload={
                        "SSIM": val_ssim,
                        "PSNR": val_psnr,
                        "MSE": val_mse,
                        "sinoMSE": val_sinomse,
                    })

                if not os.name == 'nt':
                    vessl.log(payload={"denoised_images": [
                        vessl.Image(
                            data=val_denoised_img.cpu().detach().numpy(),
                            caption=f'Epoch:{cur_epoch:4}'
                        )
                    ]})
            if cur_epoch == training_epoch - 1:
                save_network(network=network, epoch=training_epoch, optimizer=optimizer, savedir=log_dir)
                save_images(
                    val_denoised_img.cpu().detach().numpy(),
                    epoch=training_epoch,
                    tag="denoised",
                    savedir=log_dir,
                    batchnum=val_batch_size
                )
            elif cur_epoch != 0 and cur_epoch % checkpoint_intvl == 0:
                save_network(network=network, epoch=cur_epoch, optimizer=optimizer, savedir=log_dir)
                save_images(
                    val_denoised_img.cpu().detach().numpy(),
                    epoch=cur_epoch,
                    tag="denoised",
                    savedir=log_dir,
                    batchnum=val_batch_size
                )
        network.train()
        if not os.name == 'nt':
            vessl.progress((cur_epoch+1)/training_epoch)

    # End Training. Close everything
    with open(os.path.join(log_dir, 'logs.txt'), 'a') as log_file:
        print(f"Training Completed: EOF", file=log_file)