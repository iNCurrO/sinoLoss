import gdown
from config import get_config
import os
import zipfile


def main():
    url = 'https://drive.google.com/uc?id=1lb3TOWIOcJy1g5OXqfw_CIic43Jf2OJC'
    url_observerloss = 'https://drive.google.com/uc?id=1EOJMX4g1E0m3CLbf4BiPUh_RUm2N_FTK'
    print("Parse configurations")
    config = get_config()
    gdown.download(url, os.path.join(config.datadir, "downloadedfromGD"), fuzzy=True)
    gdown.download(url_observerloss, os.path.join(config.datadir, "observerloss.h5"), fuzzy=True)
    with zipfile.ZipFile(os.path.join(config.datadir, "downloadedfromGD"), 'r') as zip_ref:
        print("Start extracting... ")
        zip_ref.extractall(os.path.join(config.datadir, config.dataname))
    print("Removing zip file... ")
    os.remove(os.path.join(config.datadir, "downloadedfromGD"))


if __name__ == "__main__":
    main()
