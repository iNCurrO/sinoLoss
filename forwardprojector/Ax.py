import tigre
import numpy as np
import torch


def forwardproejection(targetimg, CTgeo):
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = 800  # Distance Source Detector      (mm)
    geo.DSO = 400  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = np.array([724, 1])  # number of pixels              (px)
    geo.dDetector = np.array([0.48828125*2, 0.48828125*2])  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = np.array([512, 512, 1])  # number of voxels              (vx)
    geo.sVoxel = np.array([250, 250, 0.48828125])  # total size of the image       (mm)
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # These two can be also defined
    # per angle

    # Auxiliary
    geo.accuracy = 0.5  # Variable to define accuracy of
    # 'interpolated' projection
    # It defines the amoutn of
    # samples per voxel.
    # Recommended <=0.5             (vx/sample)

    # Optional Parameters
    # There is no need to define these unless you actually need them in your
    # reconstruction

    geo.COR = 0  # y direction displacement for
    # centre of rotation
    # correction                   (mm)
    # This can also be defined per
    # angle

    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # X,Y and Z axis respectively. (rad)
    # This can also be defined per
    geo.mode = "cone"  # Or 'parallel'. Geometry type.

    # angle
    angles = np.linspace(0, 2*np.pi, 120)

    targetimg: np.float32 = targetimg.cpu().detach().numpy()
    batch_num = targetimg.shape[1]
    sinogram = np.array([])
    for i in range(batch_num):
        tempimg = targetimg[i, :, :, :].transpose(1, 2, 0)
        tempsino = tigre.Ax(tempimg, geo, angles)
        sinogram = sinogram.vstack([sinogram, tempsino])

    return torch.from_numpy(sinogram).cuda()
