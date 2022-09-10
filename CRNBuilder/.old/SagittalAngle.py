import matplotlib.pyplot as plt
from math import sqrt, acos, atan
import numpy as np

from Initializer import initialize


def get_sagittal_angle(Ny, Nz, vx, vy, vz, index):
    print("Computing sagittal angle...")
    # Sagittal angle computation
    #  The sagittal angle is computed as sag_ang = acos(vj / |v|), where j is the longitudinal axis along the reactor.
    #  Here j is z.
    sag_ang = np.zeros((Ny, Nz), dtype=float)
    for i in index:
        # First of all compute the modulus of velocity in each cell
        vmod = sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
        # Check if |v| == 0, in that case just force sag_ang = 0
        if vmod == 0:
            continue
        # Check if vz > |v| to avoid raising errors in acos
        if vz[i] > vmod:
            raise ValueError(f"vz < |v| in cell i: {i}\nThis should not be possible... check for errors in the code.")
        # Compute sag_ang
        sag_ang[i] = acos(vz[i] / vmod)
    print("Sagittal angle computed.")
    return sag_ang


def sagittal_angle_gradient(Ny, Nz, y, z, sag_ang, index):
    print("Computing the sagittal angle gradient...")
    sag_ang_grad_ = np.gradient(sag_ang)
    sag_ang_grad = np.zeros((Ny, Nz), dtype=float)
    for l in sag_ang_grad_:
        sag_ang_grad += np.square(l)
    sag_ang_grad = np.sqrt(sag_ang_grad)
    print("Sagittal angle gradient computed.")
    return sag_ang_grad


def get_azimuth_angle(Ny, Nz, vx, vy, index):
    print("Computing azimuth angle...")
    # Azimuth angle computation
    #  The azimuth angle is computed as azi_ang = atan(vy / vx).
    azi_ang = np.zeros((Ny, Nz), dtype=float)
    for i in index:
        # Compute azi_ang
        if vx[i] != 0:
            azi_ang[i] = atan(vy[i] / vx[i])
    print("Azimuth angle computed.")
    return azi_ang


if __name__ == '__main__':
    ncx = 3
    ncy = 3
    n_clusters = ncx * ncy

    # Load data
    case_name = "Finale_218_228"
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)

    # Get sagittal and azimuth angle
    sag_ang = get_sagittal_angle(Ny, Nz, vx, vy, vz, index)
    azi_ang = get_azimuth_angle(Ny, Nz, vx, vy, index)
    sag_ang_grad = sagittal_angle_gradient(Ny, Nz, y, z, sag_ang, index)
    sag_ang_grad_log = np.log10(1 + sag_ang_grad)

    # Plot of raw data and clusters
    print("Plotting...")
    fig = plt.figure()

    ax = fig.add_subplot(141, aspect='equal')
    cs = ax.pcolormesh(y, z, sag_ang, cmap='jet', shading='auto')
    ax.streamplot(y.T, z.T, vy.T, vz.T, density=2, linewidth=0.5, arrowsize=0.5, color='black')
    ax.set_title('Sagittal angle')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(142, aspect='equal')
    cs = ax.pcolormesh(y, z, sag_ang_grad_log, cmap='jet', shading='auto')
    ax.streamplot(y.T, z.T, vy.T, vz.T, density=2, linewidth=0.5, arrowsize=0.5, color='black')
    ax.set_title('Sagittal angle gradient')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(143, aspect='equal')
    cs = ax.pcolormesh(y, z, azi_ang, cmap='jet', shading='auto')
    ax.streamplot(y.T, z.T, vy.T, vz.T, density=2, linewidth=0.5, arrowsize=0.5, color='black')
    ax.set_title('Azimuth angle')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(144, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.streamplot(y.T, z.T, vy.T, vz.T, density=2, linewidth=0.5, arrowsize=0.5, color='black')
    ax.set_title('Temperature')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    plt.show()
