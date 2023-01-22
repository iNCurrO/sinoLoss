import os
import torch
import pickle
import numpy as np
from PIL import Image


def set_dir(config) -> str:
    if os.listdir(config.logdir):
        dirnum = int(os.listdir(config.logdir)[-1][:3])+1
    else:
        dirnum = 0
    __savedir__ = f"{dirnum:03}"
    __savedir__ = __savedir__ + "_" + str(config.model) + "_" + str(config.losses) + "_" + config.dataname
    __savedir__ = os.path.join(config.logdir, __savedir__)
    os.mkdir(__savedir__)
    return __savedir__


def save_parameters(parameters):
    pass


def save_network(network, epoch, savedir):
    saving_data = dict(network=network)
    print(f"Saving network... Dir: {savedir} // Epoch: {epoch}")
    snapshot_pkl = os.path.join(savedir, f'network-{epoch}.pkl')
    with open(snapshot_pkl, 'wb') as f:
        pickle.dump(saving_data, f)
    print(f"Save complete!")


def save_images(input_images, tag, epoch, savedir):
    print(f"Saving samples...")
    save_image_grid(input_images, os.path.join(savedir, f'samples-{epoch}-{tag}.png'))
    print(f"Save complete!\n ")


def save_image_grid(img, fname, grid_size=(1, 1)): # TODO make datasize optinize
    lo, hi = [0, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
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
        print("Error! undefined optimizer name for GAN: {}".format(config.optimizer))
        quit()
    return optimizer


def resume_network(resume):
    pass
