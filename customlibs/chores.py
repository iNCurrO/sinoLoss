import os
import torch
import pickle
import numpy as np
from PIL import Image
if not os.name == 'nt':
    import vessl


def set_dir(config):
    if not os.path.exists(config.logdir):
        print("Log directory is not exist, just we configured it\n")
        os.mkdir(config.logdir)
    logdir = os.listdir(config.logdir)
    logdir.sort()
    if os.listdir(config.logdir):
        dirnum = int(logdir[-1][:3])+1
    else:
        dirnum = 0
    __savedir__ = f"{dirnum:03}"
    losses = ""
    for i, lossname in enumerate(config.losses):
        losses += str(config.weights[i]) + lossname
    __savedir__ = __savedir__ + "_" + str(config.model) + "_" + losses + "_" + config.dataname
    if not os.name == 'nt':
        vessl.init(message=config.computername + "_" + __savedir__)
    __savedir__ = os.path.join(config.logdir, __savedir__)
    os.mkdir(__savedir__)
    return [__savedir__, dirnum]


def resume_network(resume, network, optimizer, config):
    def find_network(resume_file):
        dir_num =resume_file.split('-')[0]
        cp_num = resume_file.split('-')[1]
        try:
            logdirs = [filename for filename in os.listdir(config.logdir) if filename.startswith(dir_num)]
            if not len(logdirs) == 1:
                raise FileNotFoundError
            else:
                logdir = logdirs[0]
            fn = None
            for filename in os.listdir(os.path.join(config.logdir, logdir)):
                if filename.startswith('network-' + cp_num) and os.path.isfile(filename):
                    if fn is None:
                        fn = filename
                    else:
                        raise FileNotFoundError
            return fn
        except FileNotFoundError:
            print(f'Not founded for {resume_file}, Train with random init.\n')
            return None

    resume_file = find_network(resume_file=resume)
    if resume_file is not None:
        ckpt = torch.load(resume_file)
        network.load_state_dict(ckpt['model_state_dict'])
        if ckpt['optimizer_state_dict']:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f'Warning! there is no optimizer save file in {resume_file}, thus optimizer is going init.\n')


def save_network(network, optimizer, epoch, savedir):
    snapshot_pt = os.path.join(savedir, f'network-{epoch}.pt')
    print(f"Saving network... Dir: {savedir} // Epoch: {epoch}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, snapshot_pt)
    print(f"Save complete!")


def save_images(input_images, tag, epoch, savedir, batchnum, sino=False):
    nx = int(np.ceil(np.sqrt(batchnum)))
    save_image_grid(
        input_images,
        os.path.join(savedir, f'samples-{int(epoch):04}-{tag}.png'), grid_size=(nx, nx), sino=sino
    )


def save_image(img, fname, sino=False):
    if sino:
        lo, hi = [0, 100]
    else:
        lo, hi = [0, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    Image.fromarray(img[:, :], 'L').save(fname)


def save_image_grid(img, fname, grid_size=(1, 1), sino=False):
    if sino:
        lo, hi = [0, 100]
    else:
        lo, hi = [0, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    _N_diff = gw*gh - _N
    if _N_diff != 0:
        img = np.concatenate((img, np.zeros([_N_diff, C, H, W])))
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname)


def set_optimizer(config, model):
    if config.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learningrate, weight_decay=config.lrdecay)
    elif config.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learningrate, weight_decay=config.lrdecay)
    else:
        optimizer = None
        print("Error! undefined optimizer name for this codes: {}".format(config.optimizer))
        quit()
    return optimizer


def lprint(txt, log_dir):
    print(txt)
    with open(os.path.join(log_dir, 'logs.txt'), 'a') as log_file:
        print(txt, file=log_file)

