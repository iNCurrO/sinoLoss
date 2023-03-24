import os.path
import time
import vessl
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
    with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
        print("Start Training... \n", file=log_file)

    # Train
    val_noisy_img, val_target_img = next(iter(validation_set))
    val_batch_size = val_noisy_img.shape[0]
    save_images(val_target_img, epoch=0, tag="target", savedir=log_dir, batchnum=val_batch_size)
    save_images(val_noisy_img, epoch=0, tag='noisy', savedir=log_dir, batchnum=val_batch_size)
    network.eval()
    val_denoised_img = network(val_noisy_img.cuda())
    save_images(
        val_denoised_img.cpu().detach().numpy(), epoch=0, tag="denoised", savedir=log_dir, batchnum=val_batch_size
    )
    # save_images(
    #     loss_func.run_Amatrix(val_target_img.cuda()).cpu().detach().numpy(),
    #     epoch=0, tag="target_sino", savedir=log_dir, batchnum=val_batch_size, sino=True
    # )
    # save_images(
    #     loss_func.run_Amatrix(val_noisy_img.cuda()).cpu().detach().numpy(),
    #     epoch=0, tag='noisy_sino', savedir=log_dir, batchnum=val_batch_size, sino=True
    # ) # TODO
    network.train().requires_grad_(True)

    # Main Part
    start_time = time.time()
    print(f"Entering training at {time.localtime(start_time)}")
    for cur_epoch in range(training_epoch):
        # iteration for one epcoh
        logs = ""
        for batch_idx, samples in enumerate(training_set):
            optimizer.zero_grad()
            [noisy_img, sino, target_img] = samples
            logs = loss_func.accumulate_gradients(
                noisy_img.cuda(),
                target_img.cuda(),
                targetsino=sino.cuda()
            )
            if batch_idx % 99 == 0:
                print(
                    f'Train Epoch: {cur_epoch}/{training_epoch},' +
                    f'mean(sec/batch): {(time.time() - start_time) / cur_epoch if cur_epoch else 0}, loss:' +
                    str(logs) +
                    f'ETA: {(time.time() - start_time) / cur_epoch * (training_epoch - cur_epoch) if cur_epoch else 0}'
                )
            with torch.autograd.profiler.record_function("opt"):
                optimizer.step()
        # Print log
        print(
            f'Train Epoch: {cur_epoch}/{training_epoch},' +
            f'mean(sec/Epoch): {(time.time() - start_time) / (cur_epoch+1)}, loss:' +
            str(logs) + '\n'
        )
        with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
            print(
                f'Train Epoch: {cur_epoch}/{training_epoch},' +
                f'mean(sec/Epoch): {(time.time() - start_time) / (cur_epoch+1)}, loss:' +
                str(logs) + '\n',
                file=log_file
            )
        vessl.log(step=cur_epoch, payload={keys: logs[keys] for keys in logs})

        # Save check point and evaluate
        network.eval()
        val_denoised_img = network(val_noisy_img.cuda())
        vessl.log(payload={"denoised_images": [
            vessl.Image(
                data=val_denoised_img.cpu().detach().numpy(),
                caption=f'Epoch:{cur_epoch}'
            )
        ]})
        if cur_epoch % checkpoint_intvl == 0:
            save_network(network=network, epoch=cur_epoch, savedir=log_dir)
            save_images(
                val_denoised_img.cpu().detach().numpy(),
                epoch=cur_epoch+1,
                tag="denoised",
                savedir=log_dir,
                batchnum=val_batch_size
            )
        network.train()
        vessl.progress((cur_epoch+1)/training_epoch)
        # vessl.upload(log_dir)

    # End Training. Close everything
    with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
        with torch.autograd.profiler as prof:
            print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5), file=log_file)
        print(f"Training Completed: EOF", file=log_file)
    vessl.finish()
