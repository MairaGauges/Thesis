from functools import cached_property
from math import sqrt, pi
import matplotlib.pyplot as plt
import numpy as np

from Clustering import cluster_data_linkage
from EddyDetection import detect_eddy
from Initializer import initialize


# TODO: Non sono molto sicuro che sta roba sia così bella, forse è il caso di rifarla da capo in un modo più performante e intelligente
#  Update: In realtà è molto veloce e non sembra dare problemi di memoria, lascio il commento in caso prima o poi dia problemi ma al momento
#  sembra andare bene così

class Cell:

    def __init__(self, y, z, vy, vz, cluster):
        self.y = y
        self.z = z

        self.vy = vy
        self.vz = vz

        self.cluster = cluster

    @property
    def v(self):
        return sqrt(self.vy * self.vy + self.vz * self.vz)

    @property
    def sag_ang(self):
        angle_to_rotate = -np.arctan2(self.cluster.vz_mean, self.cluster.vy_mean)
        rotated_sag_ang = (self.vy * np.cos(angle_to_rotate) - self.vz * np.sin(angle_to_rotate), self.vy * np.sin(angle_to_rotate) + self.vz * np.cos(angle_to_rotate))
        return np.arctan2(rotated_sag_ang[1], rotated_sag_ang[0])


class Cluster:

    def __init__(self):
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)

    def remove_cell(self, cell):
        self.cells.remove(cell)

    @cached_property
    def vy_mean(self):
        vy_mean = 0.
        n = len(self.cells)
        for cell in self.cells:
            vy_mean += cell.vy
        vy_mean /= n
        return vy_mean

    @cached_property
    def vz_mean(self):
        vz_mean = 0.
        n = len(self.cells)
        for cell in self.cells:
            vz_mean += cell.vz
        vz_mean /= n
        return vz_mean

    @cached_property
    def v_mean(self):
        return sqrt(self.vy_mean * self.vy_mean + self.vz_mean * self.vz_mean)

    @cached_property
    def sag_ang(self):
        sag_ang = np.empty(len(self.cells), dtype=float)
        for i in range(len(self.cells)):
            sag_ang[i] = self.cells[i].sag_ang
        return sag_ang


def get_pxr_score(Ny, Nz, y, z, vy, vz, cluster_id, extended_output=False):
    print("Computing PXR score...")
    n_clusters = np.max(cluster_id) + 1
    pxr_score = np.zeros(n_clusters, dtype=bool)
    clusters = []
    for c in range(n_clusters):
        clusters.append(Cluster())
    for i in range(Ny):
        for j in range(Nz):
            clusters[cluster_id[i, j]].add_cell(Cell(y[i, j], z[i, j], vy[i, j], vz[i, j], clusters[cluster_id[i, j]]))
    clusters_sag_ang = []
    var_sag_ang = []
    for c in range(n_clusters):
        clusters_sag_ang.append(clusters[c].sag_ang)
        var_sag_ang.append(np.sum(np.square(clusters_sag_ang[-1] - 0)) / (clusters_sag_ang[-1].shape[0] - 1))
        if var_sag_ang[c] > pi / 2:
            pxr_score[c] = 1
    if extended_output:
        return pxr_score, clusters_sag_ang, var_sag_ang
    return pxr_score


if __name__ == '__main__':
    # Load data
    case_name = "Finale"
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)

    # Get eddy data
    # Give id 0 to cells not in an eddy, 1 to cells in the firsts eddy, 2 to the cells in the second eddy, etc...
    eddy_id = detect_eddy(case_name, Ny, Nz, y, z, vy, vz, load_cached=True)

    # Find clusters
    n_clusters = 10
    cluster_id = cluster_data_linkage(case_name, Ny, Nz, y, z, T, n_clusters, eddy_id, threshold=750, save_cache=True, to_cut=False)
    n_clusters = int(np.max(cluster_id) + 1)

    # Compute the reactor type score (0: PFR, 1: PSR)
    pxr_score, clusters_sag_ang, var_sag_ang = get_pxr_score(Ny, Nz, y, z, vy, vz, cluster_id, extended_output=True)

    # Plots
    print("Plotting results...")
    f = 1.2
    fig = plt.figure(figsize=(10 * f, 10 / 4 * 3 * f))


    def temp(n):
        for i in range(100):
            for j in range(i, int(np.floor(i * 16 / 9))):
                print(f"{i = }, {j = }, {i * j = }")
                if i * j >= n:
                    return i, j


    nx, ny = temp(n_clusters)
    for c in range(n_clusters):
        ax = fig.add_subplot(nx, ny, c + 1)
        x, _, p = ax.hist(clusters_sag_ang[c], bins=61, range=(-pi, pi))
        ax.text(pi * 0.3, max(x) * 0.90, f"{'PSR' if pxr_score[c] else 'PFR'}")
        ax.text(pi * 0.3, max(x) * 0.83, rf"$\sigma^2={var_sag_ang[c]:.2f}$", fontsize='x-small')
        ax.set_xlabel('Sagittal angle')
        # ax.set_ylabel('n° of cells')
        ax.set_xlim([-pi, pi])
        ax.set_title(f"Cluster ID: {c}")
    fig.tight_layout()

    f = 0.5
    fig = plt.figure(figsize=(8.3 * 2 * f, 8.3 * f))
    i = 1
    for c in [2, 5]:
        ax = fig.add_subplot(1, 2, i)
        i += 1
        x, _, p = ax.hist(clusters_sag_ang[c], bins=61, range=(-pi, pi))
        ax.text(pi * 0.3, max(x) * 0.90, f"{'PSR' if pxr_score[c] else 'PFR'}")
        ax.text(pi * 0.3, max(x) * 0.83, rf"$\sigma^2={var_sag_ang[c]:.2f}$", fontsize='x-small')
        ax.set_xlabel('Sagittal angle [rad]')
        ax.set_ylabel('n° of cells')
        ax.set_xlim([-pi, pi])
        ax.set_title(f"Cluster ID: {c}")
    fig.tight_layout()

    f = 0.75
    fig = plt.figure(figsize=(8.3 * f, 8.3 * f))
    ids_to_show = [2, 5]
    cluster_id_masked = np.full(cluster_id.shape, np.nan, dtype=float)
    for id_to_show in ids_to_show:
        cluster_id_masked[np.where(cluster_id == id_to_show)] = id_to_show

    ax = fig.add_subplot(121, aspect='equal')
    ax.pcolormesh(y, z, cluster_id, cmap='gray', shading='auto')
    cs = ax.pcolor(y, z, cluster_id_masked, cmap='jet', shading='auto')
    ax.set_title('Cluster ID - linkage')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    # ax.set_xlim([0, 0.02])
    # ax.set_ylim([0, 0.08])
    fig.colorbar(cs)

    ax = fig.add_subplot(122, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.set_title('Temperature')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    # ax.set_xlim([0, 0.02])
    # ax.set_ylim([0, 0.08])
    fig.colorbar(cs)
    fig.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(121, aspect='equal')
    cs = ax.pcolor(y, z, cluster_id, cmap='jet', shading='auto')
    ax.set_title('Cluster ID - linkage')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    # ax.set_xlim([0, 0.02])
    # ax.set_ylim([0, 0.08])
    fig.colorbar(cs)

    ax = fig.add_subplot(122, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.set_title('Temperature')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    # ax.set_xlim([0, 0.02])
    # ax.set_ylim([0, 0.08])
    fig.colorbar(cs)

    # nx = ny = 0
    # for i in range(100):
    #     if i ** 2 >= n_clusters:
    #         nx = ny = i
    #         break
    # for n in range(n_clusters):
    #     ax = fig.add_subplot(nx, ny, n + 1, aspect='equal')
    #     data_azi = np.zeros([], dtype=float)
    #     data_sag = np.zeros([], dtype=float)
    #     for j in range(Ny):
    #         for k in range(Nz):
    #             if cluster_id[j, k] == n:
    #                 data_azi = np.append(data_azi, azi_ang[j, k])
    #                 data_sag = np.append(data_sag, sag_ang[j, k])
    #     heatmap, xedges, yedges = np.histogram2d(data_azi, data_sag, bins=25, range=[[-pi / 2 - 0.05, pi / 2 + 0.05], [-0.05, pi + 0.05]], density=True)
    #     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #     X, Y = np.meshgrid(xedges, yedges)
    #     cs = ax.pcolormesh(X, Y, heatmap.T, cmap='hot', zorder=0)
    #     ax.text(0.8, 2.9, f"{'PSR' if pxr_score[n] > 0.5 else 'PFR'} {round(abs(pxr_score[n] - 0.5) * 200)}%", c='white', zorder=2, fontsize='x-small')
    #     ax.text(0.78, 2.73, f"{pxr_score[n]:.2f}", c='white', zorder=2, fontsize='xx-small')
    #     ax.scatter(mean_xy[n, 0], mean_xy[n, 1], color='white', marker='x', zorder=1, s=5)
    #     ax.set_xlabel('Azimuth angle')
    #     ax.set_ylabel('Sagittal angle')
    #     ax.set_xlim([-pi / 2 - 0.05, pi / 2 + 0.05])
    #     ax.set_ylim([-0.05, pi + 0.05])
    #     ax.set_title(f"Cluster ID: {n}")
    #     fig.colorbar(cs)

    plt.show()
