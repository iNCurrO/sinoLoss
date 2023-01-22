# import tigre # TODO Remove tigre option
import numpy as np
import torch


def forwardproejection(targetimg, CTgeo, use_tigre=False):
    batch_num = targetimg.shape[0]
    sinogram = torch.zeros(batch_num, 1, 724, 18)
    for i in range(batch_num):
        tempimg = targetimg[i, :, :, :].permute(1, 2, 0)
        tempsino = forward_projector(tempimg)
        sinogram[i,:,:,:] = tempsino[None, :, :]
    return sinogram.cuda()


def forward_projector(targetimg, use_tigre=False):
    if use_tigre:
        geo = tigre.geometry()
        # VARIABLE                                   DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        # Distances
        geo.DSD = 800  # Distance Source Detector      (mm)
        geo.DSO = 400  # Distance Source Origin        (mm)
        # Detector parameters
        geo.nDetector = np.array([512, 1])  # number of pixels              (px)
        geo.dDetector = np.array([0.48828125 * 2, 0.48828125 * 2])  # size of each pixel            (mm)
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
        angles = np.linspace(0, 2 * np.pi, 120)
        sinogram = tigre.Ax(targetimg, geo, angles)
    else:
        img_size = 512
        view = 18
        # view = 120
        view_angle = np.linspace(2 * np.pi / 120, 2 * np.pi, 120)
        Nx = img_size + 1
        Ny = img_size + 1
        pixel_size = 1

        X_plane = np.linspace(-img_size / 2, img_size / 2, Nx) * pixel_size
        Y_plane = np.linspace(-img_size / 2, img_size / 2, Ny) * pixel_size

        det_number = 724

        sx_temp = np.linspace(-(det_number - 1) / 2 * pixel_size, (det_number - 1) / 2 * pixel_size, det_number)
        sy_temp = -400 * np.ones(det_number)
        dx_temp = sx_temp
        dy_temp = 400 * np.ones(det_number)

        sinogram = torch.zeros((det_number, view))
        for k in range(view):
            sx, sy = rotation(sx_temp, sy_temp, view_angle[k])
            dx, dy = rotation(dx_temp, dy_temp, view_angle[k])

            for n in range(det_number):
                x1 = sx[n]
                y1 = sy[n]
                x2 = dx[n]
                y2 = dy[n]

                alpha_x = (X_plane - x1) / (x2 - x1)
                alpha_y = (Y_plane - y1) / (y2 - y1)

                X_p1 = X_plane
                Y_p1 = alpha_x * (y2 - y1) + y1
                Y_p2 = Y_plane
                X_p2 = alpha_y * (x2 - x1) + x1

                X_p = np.concatenate((X_p1, X_p2))
                Y_p = np.concatenate((Y_p1, Y_p2))

                index = np.where((X_p >= X_plane[0]) & (X_p <= X_plane[-1]) & (Y_p >= Y_plane[0]) & (Y_p <= Y_plane[-1]))

                X_p = X_p[index]
                Y_p = Y_p[index]
                point_array = np.vstack((X_p, Y_p))
                point_array = point_array[:, np.argsort(point_array[0, :])]
                X_p = point_array[0, :]
                Y_p = point_array[1, :]

                distance = np.sqrt((X_p[:-1] - X_p[1:]) ** 2 + (Y_p[:-1] - Y_p[1:]) ** 2)
                null_index = np.where(distance == 0)
                distance = np.delete(distance, null_index)
                X_p = np.delete(X_p, null_index)
                Y_p = np.delete(Y_p, null_index)

                col_index = np.ceil((X_p[:-1] + X_p[1:]) / 2 / pixel_size + (img_size / 2)) - 1
                row_index = img_size - np.ceil(((Y_p[:-1] + Y_p[1:]) / 2 / pixel_size + img_size / 2))
                temp_weight = np.zeros(targetimg.shape)
                for idx, d in enumerate(distance):
                    temp_weight[int(row_index[idx]), int(col_index[idx])] = d
                sinogram[n, k] = torch.dot(torch.flatten(targetimg.cpu()), torch.flatten(torch.from_numpy(np.float32(temp_weight))))
                # sinogram[n, k] = np.convolve(np.reshape(targetimg, targetimg.size), np.reshape(temp_weight, temp_weight.size), 'valid')
    return sinogram


def rotation(x, y, theta):
    rx = x * np.cos(theta) - y * np.sin(theta)
    ry = x * np.sin(theta) + y * np.cos(theta)
    return rx, ry
