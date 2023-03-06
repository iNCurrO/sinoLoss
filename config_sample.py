import argparse
import config

arg_lists = []
parser = argparse.ArgumentParser()


# Config list
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batchsize', type=int, default=4, help='Number of batch size. Recommend power of 2')
data_arg.add_argument('--valbatchsize', type=int, default=1, help='Number of batch size. Must be square of int')
data_arg.add_argument('--datadir', type=str, default="G:\CT\Image")
data_arg.add_argument('--dataname', type=str, default="SpraseView_parallel")


# Network
network_arg = add_argument_group('Network')
network_arg.add_argument('--model', type=str, default="unet", choices=["UNET", 'VGG'])
network_arg.add_argument('--losses', type=tuple, default=tuple(["MSE", "sinoloss_MSE"]), ) # ['MSE', 'MAE', 'sinoloss_MSE', 'sinoloss_MAE']
network_arg.add_argument('--weights', type=tuple, default=tuple([0.7, 0.3]),)
network_arg.add_argument('--resume', type=str, default=None)


# Forward Projection
ForwardProj = add_argument_group('ForwardProejction')
ForwardProj.add_argument('--img_size', type=list, default=[512, 512],
                    help='Phantom image size')
ForwardProj.add_argument('--pixel_size', type=float, default=1,
                    help='Pixel size of the phantom image')
ForwardProj.add_argument('--quarter_offset', action='store_true',
                    help='detector quarter offset')
ForwardProj.add_argument('--geometry', type=str, default='parallel',
                    help='CT geometry')
ForwardProj.add_argument('--mode', type=str, default='equiangular',
                    help='CT detector arrangement')
ForwardProj.add_argument('--view', type=int, default=18,
                    help='number of view (should be even number for quarter-offset')
ForwardProj.add_argument('--num_split', type=int, default=256,
                    help='number of splitting processes for FP: fewer number guarantee faster speed by reducing number of loop but memory consuming')
ForwardProj.add_argument('--datatype', type=str, default='float',
                    help='datatype of tensor: double type guarantee higher accuracy but memory consuming (double: 64bit, float: 32bit)')
ForwardProj.add_argument('--cpu', action='store_true',
                    help='Activate CPU mode')

# Geometry conditions
ForwardProj.add_argument('--SCD', type=float, default=400,
                    help='source-center distance (mm scale)')
ForwardProj.add_argument('--SDD', type=float, default=800,
                    help='source-detector distance (mm scale)')
ForwardProj.add_argument('--num_det', type=int, default=724,
                    help='number of detector')
ForwardProj.add_argument('--det_interval', type=float, default=1,
                    help='interval of detector (mm scale)')
ForwardProj.add_argument('--det_lets', type=int, default=1,
                    help='number of detector lets')


# System parameters
sysparm_arg = add_argument_group('System')
sysparm_arg.add_argument('--logdir', type=str, default='G:\CT\sinoloss logs')
sysparm_arg.add_argument('--trainingepoch', type=int, default=200)
sysparm_arg.add_argument('--optimizer', type=str, default="ADAM", choices=["ADAM", "ADAMW"])
sysparm_arg.add_argument('--learningrate', type=float, default=0.0001)
sysparm_arg.add_argument('--lrdecay', type=float, default=0)
sysparm_arg.add_argument('--numworkers', type=int, default=2)
sysparm_arg.add_argument('--training', type=bool, default=True)
sysparm_arg.add_argument('--save_intlvl', type=int, default=1)
sysparm_arg.add_argument('--debugging', type=bool, default=False)
sysparm_arg.add_argument('--hyperrecord', type=bool, default=False)

# # CT geometry parameters
# geometry_arg = add_argument_group('Geometry')
# geometry_arg.add_argument('--DSD', type=float, default=800)
# geometry_arg.add_argument('--DSO', type=float, default=400)
# geometry_arg.add_argument('--nDet', type=int, default=724)#182)#724)
# geometry_arg.add_argument('--dDet', type=float, default=1)
# geometry_arg.add_argument('--nView', type=int, default=18)
# geometry_arg.add_argument('--nPixel', type=int, default=512)#128)#512)
# geometry_arg.add_argument('--dPixel', type=float, default=1)
# geometry_arg.add_argument('--mode', type=str, default='parallel')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config
