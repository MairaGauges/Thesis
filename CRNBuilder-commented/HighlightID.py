import sys

import matplotlib.pyplot as plt
import numpy as np
import os

from Initializer import initialize


# This script is used to generate a plot in which only some clusters (selected based on the id) are colored, while all the others are left in grey-scale

def main():
    # Load data
    case_name = "Finale"
    if sys.argv[1]:
        cluster_number = int(sys.argv[1])
    else:
        cluster_number = 19
    ids_to_show = [i for i in range(18)]
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)

    cluster_id = np.array(np.load(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"cluster_ids", f"cluster_id_{cluster_number}cl.npy")), dtype=float)
    cluster_id_masked = np.full(cluster_id.shape, np.nan, dtype=float)
    for id_to_show in ids_to_show:
        cluster_id_masked[np.where(cluster_id == id_to_show)] = id_to_show

    f = 1.5
    fig = plt.figure(figsize=(8.3 / 1.5 * f, 8.3 / 1.5 / 1.5 * f))

    ax = fig.add_subplot(121, aspect='equal')
    ax.pcolormesh(y, z, cluster_id, cmap='gray', shading='auto')
    cs = ax.pcolor(y, z, cluster_id_masked, cmap='jet', shading='auto')
    ax.set_title('Cluster IDs')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    ax = fig.add_subplot(122, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.set_title('Temperature')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    fig.colorbar(cs)

    plt.show()


if __name__ == '__main__':
    main()
