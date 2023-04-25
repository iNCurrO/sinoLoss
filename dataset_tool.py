import gdown
from config import get_config
import os
import zipfile


def main():
    url = 'https://drive.google.com/uc?id=19lGLrBsIZfUfhOaGTv29cbWRGTiX50_z'
    url_observerloss = 'https://drive.google.com/uc?id=1EOJMX4g1E0m3CLbf4BiPUh_RUm2N_FTK'
    print("Parse configurations")
    config = get_config()
    if not os.path.exists(os.path.join(config.datadir)):
        os.mkdir(os.path.join(config.datadir))
    gdown.download(url, os.path.join(config.datadir, "downloadedfromGD"), fuzzy=True)
    gdown.download(url_observerloss, os.path.join(config.datadir, "observerloss.h5"), fuzzy=True)
    with zipfile.ZipFile(os.path.join(config.datadir, "downloadedfromGD"), 'r') as zip_ref:
        print("Start extracting... ")
        zip_ref.extractall(os.path.join(config.datadir, config.dataname))
    print("Removing zip file... ")
    os.remove(os.path.join(config.datadir, "downloadedfromGD"))


if __name__ == "__main__":
    main()
