import argparse

arg_lists = []
parser = argparse.ArgumentParser()


# Config list
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batchsize', type=int, default=2, help='Number of batch size. Recommend power of 2')
data_arg.add_argument('--datadir', type=str, default="D:\CTdata\SpraseView_parallel")
data_arg.add_argument('--dataname', type=str, default="sinoloss_f_SV")
data_arg.add_argument('--FVdataname', type=str, default="sinoloss_f_FV")


# Network
network_arg = add_argument_group('Network')
network_arg.add_argument('--model', type=str, default="unet", choices=["UNET", 'VGG'])
network_arg.add_argument('--losses', type=tuple, default=tuple(["MSE"]), ) # ['MSE', 'MAE', 'sinoloss_MSE', 'sinoloss_MAE']
network_arg.add_argument('--weights', type=tuple, default=tuple([1.]),)
network_arg.add_argument('--resume', type=str, default=None)

# System parameters
sysparm_arg = add_argument_group('System')
sysparm_arg.add_argument('--logdir', type=str, default='./logs')
sysparm_arg.add_argument('--trainingepoch', type=int, default=40)
sysparm_arg.add_argument('--optimizer', type=str, default="ADAM", choices=["ADAM", "ADAMW"])
sysparm_arg.add_argument('--learningrate', type=float, default=0.0001)  # 0.0001 for CNN-observer 0.005 for GAN
sysparm_arg.add_argument('--lrdecay', type=float, default=0)
sysparm_arg.add_argument('--numworkers', type=int, default=4)
sysparm_arg.add_argument('--training', type=bool, default=True)
sysparm_arg.add_argument('--debugging', type=bool, default=False)
sysparm_arg.add_argument('--hyperrecord', type=bool, default=False)

# CT geometry parameters
geometry_arg = add_argument_group('Geometry')
geometry_arg.add_argument('--DSD', type=float, default=800)
geometry_arg.add_argument('--DSO', type=float, default=400)
geometry_arg.add_argument('--nDet', type=int, default=724)
geometry_arg.add_argument('--dDet', type=float, default=0.48828125*2)
geometry_arg.add_argument('--nPixelX', type=int, default=512)
geometry_arg.add_argument('--nPixelY', type=int, default=512)
geometry_arg.add_argument('--dPixelX', type=float, default=0.48828125)
geometry_arg.add_argument('--dPixelY', type=float, default=0.48828125)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config
