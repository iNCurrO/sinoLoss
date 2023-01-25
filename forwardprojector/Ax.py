# import tigre # TODO Remove tigre option
import numpy as np
import torch


def forwardproejection(targetimg, config):
    batch_num = targetimg.shape[0]
    sinogram = torch.zeros(batch_num, 1, config.nDet, config.nView)
    for i in range(batch_num):
        tempimg = targetimg[i, :, :, :].permute(1, 2, 0)
        tempsino = forward_projector(tempimg, config)
        sinogram[i, :, :, :] = tempsino[None, :, :]
    return sinogram.cuda()


def forward_projector(targetimg, config):
    img_size = config.nPixel
    view = config.nView
    view_angle = np.linspace(2 * np.pi / view, 2 * np.pi, view)
    Nx = img_size + 1
    Ny = img_size + 1
    pixel_size = config.dPixel

    X_plane = np.linspace(-img_size / 2, img_size / 2, Nx) * pixel_size
    Y_plane = np.linspace(-img_size / 2, img_size / 2, Ny) * pixel_size

    det_number = config.nDet
    det_size = config.dDet

    sx_temp = np.linspace(-(det_number - 1) / 2 * det_size, (det_number - 1) / 2 * det_size, det_number)
    sy_temp = (config.DSO-config.DSD) * np.ones(det_number)
    dx_temp = sx_temp
    dy_temp = config.DSO * np.ones(det_number)

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
    return sinogram


def rotation(x, y, theta):
    rx = x * np.cos(theta) - y * np.sin(theta)
    ry = x * np.sin(theta) + y * np.cos(theta)
    return rx, ry
