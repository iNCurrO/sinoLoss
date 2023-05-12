import gdown
from config import get_config
import os
import zipfile


def main():
    url = 'https://drive.google.com/uc?id=19lGLrBsIZfUfhOaGTv29cbWRGTiX50_z'
    url_observerloss = 'https://drive.google.com/uc?id=1EOJMX4g1E0m3CLbf4BiPUh_RUm2N_FTK'
    url_sparview_fan_noisy = 'https://drive.google.com/file/d/1-CoXMzCKO5y5mqDIoQDmHEgT2KL8r1pT/view?usp=sharing'
    print("Parse configurations")
    config = get_config()
    if not os.path.exists(os.path.join(config.datadir)):
        os.mkdir(os.path.join(config.datadir))
    gdown.download(url_observerloss, os.path.join(config.datadir, "observerloss.h5"), fuzzy=True)
    if not os.path.exists(os.path.join(config.datadir, "SparseView_fan")):
        gdown.download(url, os.path.join(config.datadir, "downloadedfromGD_fan"), fuzzy=True)
        extract_and_remove("downloadedfromGD_fan", "SparseView_fan", config)
    if not os.path.exists(os.path.join(config.datadir, "SparseView_fan_noisy")):
        gdown.download(url_sparview_fan_noisy, os.path.join(config.datadir, "downloadedfromGD_fan_noisy"), fuzzy=True)
        extract_and_remove("downloadedfromGD_fan_noisy", "SparseView_fan_noisy", config)

def extract_and_remove(filename, targetdir, config):
    with zipfile.ZipFile(os.path.join(config.datadir, filename), 'r') as zip_ref:
        print("Start extracting... ")
        zip_ref.extractall(os.path.join(config.datadir, targetdir))
    print("Removing zip file... ")
    os.remove(os.path.join(config.datadir, filename))


if __name__ == "__main__":
    main()
