from __future__ import annotations  # This is used to enable type hinting in python versions from 3.7 to 3.9 (it is built-in in 3.10 and above)

import os.path
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import sparse
import scipy.sparse.linalg as linalg
from time import time

from Initializer import initialize
from InputParser import parse_input
from EddyDetection import detect_eddy


def normalize(x: np.array[float], ymin: float = 0, ymax: float = 1) -> np.array[float]:
    """
    Used to normalize a numpy array of floats between ymin and ymax (default between 0 and 1)
    :param x: Numpy array to be normalized
    :param ymin: Min value of the normalized array, default to 0
    :param ymax: Max value of the normalized array, default to 1
    :return: The normalized array
    """
    # print("Normalizing...")
    xmin = np.max(x)
    xmax = np.min(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < xmin:
                xmin = x[i, j]
            if x[i, j] > xmax:
                xmax = x[i, j]

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if xmin <= x[i, j] <= xmax:
                x[i, j] -= xmin
                x[i, j] /= (xmax - xmin)
            x[i, j] *= (ymax - ymin)
            x[i, j] += ymin
    return x


def cluster_data_kmeans(Ny, Nz, y, z, T, index, n_clusters):
    """
    !!! OBSOLETE !!!
    This operates a k-mean clustering based on the y and z coordinates and Temperature of each cell
    :param Ny:
    :param Nz:
    :param y:
    :param z:
    :param T:
    :param index:
    :param n_clusters:
    :return:
    """
    print("------------- k-means clustering -------------")
    print("Starting clustering phase...")

    # Normalize
    T = T.copy()
    T = normalize(T)
    y = y.copy()
    y = normalize(y, 0, np.max(y) / (np.max(z) - np.min(z)))
    z = z.copy()
    z = normalize(z)

    # Clustering phase
    # Organize data
    data = np.zeros([len(index), 3], dtype=float)
    for i in range(len(index)):
        data[i, 0] = y[index[i]]
        data[i, 1] = z[index[i]]
        data[i, 2] = T[index[i]]
        if isnan(T[index[i]]):
            print(f"i: {i}, T: {T[index[i]]}")
    # Execute k-means
    centroid, label = kmeans2(data, n_clusters, iter=1000, minit='points')
    print(f"Found {len(centroid)} clusters.")
    print("Centroid data:")
    print("x_mean       y_mean       T_mean")
    for c in range(centroid.shape[0]):
        print(f"{centroid[c, 0]:.4f}       {centroid[c, 1]:.4f}       {centroid[c, 2]:.4f}")
    # Get data from label
    cluster_id = np.zeros([Ny, Nz], dtype=int)
    for i in range(len(index)):
        cluster_id[index[i]] = label[i] + 1

    return cluster_id


def cluster_data_spectral(Ny, Nz, y, z, T, index, n_clusters):
    """
    !!! OBSOLETE !!!
    This operates a spectral clustering based on the y and z coordinates and Temperature of each cell
    """
    print("------------- Spectral clustering ------------")
    print("Starting clustering phase...")

    # Normalize
    T = T.copy()
    T = normalize(T)
    y = y.copy()
    y = normalize(y, 0, np.max(y) / (np.max(z) - np.min(z)))
    z = z.copy()
    z = normalize(z)

    # Clustering phase
    # Initialize the weighted adjacency matrix
    print("Initializing weight matrix...")
    data = []
    index_i = []
    index_j = []
    for i in range(Nz):
        for j in range(Ny):
            if i > 0:
                data.append(1 - abs(T[i, j] - T[i - 1, j]))
                index_i.append(j + i * Ny)
                index_j.append(j + (i - 1) * Ny)
            if j > 0:
                data.append(1 - abs(T[i, j] - T[i, j - 1]))
                index_i.append(j + i * Ny)
                index_j.append(j - 1 + i * Ny)
            if i < Nz - 1:
                data.append(1 - abs(T[i, j] - T[i + 1, j]))
                index_i.append(j + i * Ny)
                index_j.append(j + (i + 1) * Ny)
            if j < Ny - 1:
                data.append(1 - abs(T[i, j] - T[i, j + 1]))
                index_i.append(j + i * Ny)
                index_j.append(j + 1 + i * Ny)
    weight = (sparse.coo_matrix((data, (index_i, index_j)), shape=(Ny * Nz, Ny * Nz), dtype=float)).tocsc()

    # Initialize the Laplacian matrix
    # First create the diagonal matrix of node degrees
    print("Initializing diagonal matrix...")
    data = []
    index_i = []
    index_j = []
    for i in range(Nz):
        for j in range(Ny):
            data.append(np.sum(weight[j + i * Ny, :]))
            index_i.append(j + i * Ny)
            index_j.append(j + i * Ny)
            if data[-1] == 0:
                raise ValueError(f"The cell (i: {i}, j: {j}) is not connected to any cell in the graph.\n"
                                 f"This is mostly due to the fact that its value is max(T) and the neighboring cells "
                                 f"have value min(T).")
    diagonal = (sparse.coo_matrix((data, (index_i, index_j)), shape=(Ny * Nz, Ny * Nz), dtype=float)).tocsc()

    # Delete unnecessary lists to save memory
    del data
    del index_i
    del index_j

    # Compute the Laplacian and delete the other 2 matrices to save memory
    print("Computing laplacian matrix...")
    laplacian = diagonal - weight

    del diagonal
    del weight

    print("Finding eigenvalues...")
    w, v = linalg.eigsh(laplacian, k=n_clusters + 1, sigma=0)
    v = v[:, 1:]

    print("Running k-means...")
    centroid, label = kmeans2(v, 2 ** n_clusters, iter=1000, minit='++')

    print(f"Found {len(centroid)} clusters.")
    # print("Centroid data:")
    # print("x_mean       y_mean       T_mean")
    # for c in range(centroid.shape[0]):
    #     print(f"{centroid[c, 0]:.4f}       {centroid[c, 1]:.4f}       {centroid[c, 2]:.4f}")
    # Get data from label
    cluster_id = np.zeros([Ny, Nz], dtype=int)
    for i in range(len(index)):
        cluster_id[index[i]] = label[i] + 1

    return cluster_id


class Cluster:

    def __init__(self, cl_map: ClusterMap, id_: int, starting_cell: list[float]):
        self.id_ = id_
        self.cells = np.array([starting_cell], dtype=int)
        self.neighbors = []

        self.cl_map = cl_map

        self.min_diff = np.inf
        self.target = None

        self.empty = False

        self.max_s = 0
        self.min_s = np.inf
        for cell in self.cells:
            if self.cl_map.s[cell[0], cell[1]] > self.max_s:
                self.max_s = self.cl_map.s[cell[0], cell[1]]
            if self.cl_map.s[cell[0], cell[1]] < self.min_s:
                self.min_s = self.cl_map.s[cell[0], cell[1]]

    def merge(self, other_cluster: Cluster):
        """
        Add to this cluster the data of other_cluster and check other_cluster as empty
        :param other_cluster: Cluster object
        :return: None
        """
        last_index = self.cells.shape[0] - 1
        # Add the cells of the other cluster to this cluster
        self.add_cells(other_cluster.cells)
        # Cycle through the added cells to check if max_s or min_s should be modified
        for i in range(last_index + 1, self.cells.shape[0]):
            if self.cl_map.s[self.cells[i][0], self.cells[i][1]] > self.max_s:
                self.max_s = self.cl_map.s[self.cells[i][0], self.cells[i][1]]
            if self.cl_map.s[self.cells[i][0], self.cells[i][1]] < self.min_s:
                self.min_s = self.cl_map.s[self.cells[i][0], self.cells[i][1]]
        # Add the other cluster's neighbors to this cluster and add this cluster as a neighbor
        for n in other_cluster.neighbors:
            if n.id_ != self.id_:
                self.neighbors.append(n)
                n.neighbors.append(self)
        # Check the other cluster as empty
        other_cluster.clear()

    def add_cells(self, cells: np.array[float]):
        """
        Add cells to the cells array of the current cluster
        :param cells: The cells to be added
        :return: None
        """
        self.cells = np.vstack((self.cells, cells))

    def diff_from_cluster(self, other_cluster: Cluster) -> float:
        """
        Get the maximum difference between this cluster and other_cluster
        :param other_cluster: Cluster object
        :return: float
        """
        # TODO: Add caching to improve performance (this function is one of the bottlenecks of the program)
        return max(self.max_s - other_cluster.min_s,
                   other_cluster.max_s - self.min_s)

    def get_min_diff(self):
        """
        Get the least difference between this cluster and its neighbors
        :return: None
        """
        for n in self.neighbors:
            d = self.diff_from_cluster(n)
            if d < self.min_diff:
                self.target = n
                self.min_diff = d

    def get_target_under_threshold(self):
        self.target = None
        if self.max_s > self.cl_map.threshold:
            return
        for n in self.neighbors:
            if n.max_s <= self.cl_map.threshold:
                self.target = n
                break

    def get_target_same_cluster_id(self, cluster_id):
        self.target = None
        for n in self.neighbors:
            if cluster_id[n.cells[0][0], n.cells[0][1]] == cluster_id[self.cells[0][0], self.cells[0][1]]:
                self.target = n
                break

    def reset_min_diff(self):
        """
        Reset min_diff to infinity
        :return: None
        """
        self.min_diff = np.inf

    def clean_neighbors(self):
        """
        Remove empty clusters from the neighbors list
        :return: None
        """
        self.neighbors = [n for n in self.neighbors if not n.empty]

    def clear(self):
        """
        Set this cluster as empty and remove its data
        :return: None
        """
        self.cells = np.array([], dtype=int)
        self.neighbors = []
        self.min_diff = np.inf
        self.target = None
        self.empty = True

    @property
    def s_range(self):
        return self.max_s - self.min_s

    @property
    def s_mean(self):
        s_mean = 0.
        for cell in self.cells:
            s_mean += self.cl_map.s[cell[0], cell[1]]
        s_mean /= len(self.cells)
        return s_mean


class ClusterMap:

    def __init__(self, Ny: int, Nz: int, s: np.array[float], eddy_id: np.array[float], n_clusters, threshold):
        # Save the scalar field inside this object to be accessed by all the clusters
        self.s = s
        self.epsilon = 0.01  # TODO: Add this parameter to the input file
        # Initialize a grid of one cluster per cell
        self.clusters = [Cluster(self, i * Nz + j, [i, j]) for i in range(Ny) for j in range(Nz)]
        # Set the neighbors of each cluster (theoretically by not connecting clusters from different eddies as each eddy should remain in its own cluster)
        for i in range(Ny):
            for j in range(Nz):
                # Python is written in such a way that parameters are passed as reference if the passed variable
                # is mutable (here, since Cluster is an object, it is mutable)
                if i > 0 and eddy_id[i, j] == eddy_id[i - 1, j]:
                    self.clusters[i * Nz + j].neighbors.append(self.clusters[(i - 1) * Nz + j])
                if i < Ny - 1 and eddy_id[i, j] == eddy_id[i + 1, j]:
                    self.clusters[i * Nz + j].neighbors.append(self.clusters[(i + 1) * Nz + j])
                if j > 0 and eddy_id[i, j] == eddy_id[i, j - 1]:
                    self.clusters[i * Nz + j].neighbors.append(self.clusters[i * Nz + (j - 1)])
                if j < Nz - 1 and eddy_id[i, j] == eddy_id[i, j + 1]:
                    self.clusters[i * Nz + j].neighbors.append(self.clusters[i * Nz + (j + 1)])
        self.n_clusters = n_clusters
        self.threshold = threshold

    def merge_from_cache(self, cluster_id):
        """
        Cycle through all the clusters in the grid, search for the ones with the same cluster_id value and
        merge them
        :return: bool
        """
        for cluster in self.clusters:
            if cluster.empty:
                continue
            cluster.clean_neighbors()
            cluster.get_target_same_cluster_id(cluster_id)
        all_None = True
        for cluster in self.clusters:
            if cluster.empty or (cluster.target is None) or cluster.target.empty:
                continue
            all_None = False
            cluster.merge(cluster.target)
        return all_None

    def merge_under_threshold(self):
        """
        Cycle through all the clusters in the grid, search for the ones with the Temperature under the threshold and
        merge them
        :return: bool
        """
        for cluster in self.clusters:
            if cluster.empty:
                continue
            cluster.clean_neighbors()
            cluster.get_target_under_threshold()
        all_None = True
        for cluster in self.clusters:
            if cluster.empty or (cluster.target is None) or cluster.target.empty:
                continue
            all_None = False
            cluster.merge(cluster.target)
        return all_None

    def find_and_merge(self):
        """
        Cycle through all the clusters in the grid, search for the ones with the least difference in temperature and
        merge them
        :return: None
        """
        # Reset clusters, clean them and prepare them for the algorithm
        for cluster in self.clusters:
            if cluster.empty:
                continue
            cluster.reset_min_diff()
            cluster.clean_neighbors()
            cluster.get_min_diff()
        # Create the difference array and search for the clusters to merge
        diff = np.array([cluster.min_diff for cluster in self.clusters], dtype=float)
        if self.get_cluster_number() > self.n_clusters * 10:
            min_indices = np.flatnonzero(diff <= diff.min() * (1 + self.epsilon))
        else:
            min_indices = np.flatnonzero(diff <= diff.min())
        # Cycle through the clusters to be merged
        for index in min_indices:
            # Check if the cluster still exists
            if self.clusters[index].empty or (self.clusters[index].target is None) or self.clusters[index].target.empty:
                continue
            self.clusters[index].merge(self.clusters[index].target)

    def cut(self, y, z, max_ratio=0.3):
        """
        !!! OBSOLETE !!!
        This function has proven to be not efficient in improving the results of the program
        :param y: ...
        :param z: ...
        :param max_ratio: Maximum ratio of the length of a single cluster with respect to the longest dimension of the
                          original grid
        :return: None
        """
        # TODO: Optimize this "thing". For real, it's awful to look at, moreover it doesn't keep geometric consistency
        #  (i.e. it happens that a cluster is not continuous)
        #  Maybe check for cluster length directly during the clustering phase? Add a cleanup instead of a cut in the end?
        #  Update: This function is not used anymore, I'll leave the comment there just in case anyone wants to look at it
        #  but it should not be necessary
        # Search for the longest dimension of the grid
        length = max(np.max(z) - np.min(z), np.max(y) - np.min(y))
        # Start by cleaning up self.clusters of all the empty cluster
        self.clusters = [cluster for cluster in self.clusters if not cluster.empty]
        for i in range(100):
            to_cut = False
            for cluster in self.clusters:
                y_min = np.inf
                y_max = -np.inf
                z_min = np.inf
                z_max = -np.inf
                for cell in cluster.cells:
                    if y[cell[0], cell[1]] > y_max:
                        y_max = y[cell[0], cell[1]]
                    if y[cell[0], cell[1]] < y_min:
                        y_min = y[cell[0], cell[1]]
                    if z[cell[0], cell[1]] > z_max:
                        z_max = z[cell[0], cell[1]]
                    if z[cell[0], cell[1]] < z_min:
                        z_min = z[cell[0], cell[1]]
                cluster_length_y = y_max - y_min
                cluster_length_z = z_max - z_min
                if cluster_length_y > max_ratio * length:
                    to_cut = True
                    mean_y = y_min + (y_max - y_min) / 2
                    new_cluster = Cluster(self, 0, [0, 0])
                    to_keep = np.ones(cluster.cells.shape[0], dtype=bool)
                    for i in range(len(cluster.cells)):
                        if y[cluster.cells[i, 0], cluster.cells[i, 1]] > mean_y:
                            new_cluster.add_cells(cluster.cells[i, :])
                            to_keep[i] = False
                    cluster.cells = cluster.cells[to_keep, :]
                    self.clusters.append(new_cluster)
                    new_cluster.cells = new_cluster.cells[1:, :]
                if cluster_length_z > max_ratio * length:
                    to_cut = True
                    mean_z = z_min + (z_max - z_min) / 2
                    new_cluster = Cluster(self, 0, [0, 0])
                    to_keep = np.ones(cluster.cells.shape[0], dtype=bool)
                    for i in range(len(cluster.cells)):
                        if z[cluster.cells[i, 0], cluster.cells[i, 1]] > mean_z:
                            new_cluster.add_cells(cluster.cells[i, :])
                            to_keep[i] = False
                    cluster.cells = cluster.cells[to_keep, :]
                    self.clusters.append(new_cluster)
                    new_cluster.cells = new_cluster.cells[1:, :]
            if not to_cut:
                break

    def rework(self, Ny, Nz):
        """
        !!! OBSOLETE !!!
        This function has proven to be not efficient in improving the results of the program, it merges clusters with an area less than
        1% of the total area of the system
        :param Ny: ...
        :param Nz: ...
        :return: None
        """
        total_area = Ny * Nz
        cleaned = True
        while cleaned:
            cleaned = False
            for cluster in self.clusters:
                if cluster.empty:
                    continue
                cluster_area = len(cluster.cells)
                if 0 < cluster_area < 0.01 * total_area:
                    cluster.reset_min_diff()
                    cluster.clean_neighbors()
                    cluster.get_min_diff()
                    if len(cluster.target.cells) > 0 and cluster.min_diff < 100.:
                        cluster.merge(cluster.target)
                        cleaned = True
                        break

    def get_cluster_number(self) -> int:
        """
        Return the number of non-empty clusters
        :return: Non-empty clusters number
        """
        num = 0
        for cluster in self.clusters:
            if not cluster.empty:
                num += 1
        return num


def get_n_cl_cache(path: str, n_clusters: int):
    """
    This function is used to search the lowest number of clusters present in the cache still above the objective number
    e.g. if there are files in the cache storing the configurations for 40, 50 and 60 clusters and the objective number is 45, 50 is returned.
    :param path: Path in which the cached files are stored
    :param n_clusters: Objective number of clusters
    :return: lowest number of clusters present in the cache still above the objective number
    """
    n_cl_cache = -1
    for file in os.listdir(path):
        if len(file.split("_")) > 2:
            for i in range(len(file.split("_")[2])):
                if file.split("_")[2][i] == "c" and int(file.split("_")[2][:i]) > n_cl_cache and int(file.split("_")[2][:i]) > n_clusters:
                    n_cl_cache = int(file.split("_")[2][:i])
                    return n_cl_cache
    return n_cl_cache


def cluster_data_linkage(case_name: str, Ny: int, Nz: int,
                         y: np.array[float], z: np.array[float],
                         s: np.array[float], n_clusters: int,
                         eddy_id: np.array[int], threshold: float,
                         load_cached: bool = True, save_cache: bool = True,
                         to_cut: bool = False, rework: bool = True) -> np.array[int]:
    """
    Cluster data based on Temperature using a linkage-like algorithm that checks for adjacency of cells to be clustered
    :param case_name: Case name used for caching purposes
    :param Ny: Number of cells along the radial/secondary direction
    :param Nz: Number of cells along the axial/primary direction
    :param y: Matrix of the coordinates of each cell along the radial/secondary direction
    :param z: Matrix of the coordinates of each cell along the axial/primary direction
    :param s: Matrix of the scalar field on which the clustering is based (i.e. at the current state of the program, Temperature)
    :param n_clusters: Number of cluster to divide the scalar field, before the cut along y and z
    :param eddy_id: ...
    :param threshold: ...
    :param load_cached: If True search for cached data and load it, defaults to True
    :param save_cache: If True save the obtained cluster_id array, defaults to True
    :param to_cut: ...
    :param rework: ...
    :return: 2D numpy array of the id of the cluster to which each cell is assigned
    """
    print("------------- Linkage clustering -------------")
    print("Starting clustering phase...")

    # Load cached data if present for the objective number of clusters
    if load_cached and os.path.isfile(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"cluster_ids", f"cluster_id_{n_clusters}cl.npy")):
        print(f"Found cached data. Loading...")
        cluster_id = np.load(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"cluster_ids", f"cluster_id_{n_clusters}cl.npy"))
        print(f"Total cluster number: {int(np.max(cluster_id) + 1)}")
        return cluster_id
    print("Cached data not found, generating clusters...")
    print(f"Target number of clusters: {n_clusters}")

    # Normalize the input fields
    threshold -= np.min(s)
    threshold /= (np.max(s) - np.min(s))
    s = s.copy()
    s = normalize(s)
    y = y.copy()
    y = normalize(y, 0, np.max(y) / (np.max(z) - np.min(z)))
    z = z.copy()
    z = normalize(z)

    # Initialize the cluster map that will operate the clustering
    cluster_map = ClusterMap(Ny, Nz, s, eddy_id, n_clusters, threshold)

    # Maximum number of iteration to avoid infinite loops
    max_iter = cluster_map.get_cluster_number()

    start_time = time()
    print("Starting the clustering process... (This may take a long time)")
    print(" [%]    Total number of clusters")
    initial_number = cluster_map.get_cluster_number() + 1
    # Search for cached data that can be useful in speeding up the algorithm (by providing a starting point with less clusters)
    if os.path.isdir(os.path.join(os.getcwd(), f"data", f"{case_name}", f"cache", f"cluster_ids")):
        path = os.path.join(os.getcwd(), f"data", f"{case_name}", f"cache", f"cluster_ids")
        n_cl_cache = get_n_cl_cache(path, n_clusters)
        if n_cl_cache > 0:
            print(f"Found a cached cluster_id with {n_cl_cache} clusters, loading it and using it as a starting point...")
            cluster_id = np.load(os.path.join(f"{path}", f"cluster_id_{n_cl_cache}cl.npy"))
            for i in range(max_iter):
                print(f"\r{(initial_number - (cluster_map.get_cluster_number() - n_clusters + 1)) / initial_number * 100:>5.2f}   "
                      f"{cluster_map.get_cluster_number():<24} (Pre-processing phase - merging from id)", end="")
                if cluster_map.merge_from_cache(cluster_id):
                    break
                if i == max_iter - 1:
                    raise RuntimeError(f"Could not finish the pre-clustering algorithm within {max_iter} iterations, try changing "
                                       f"max_iter in the function cluster_data_linkage() inside Clustering.py or search for the "
                                       f"bug that prevents the algorithm from converging.")
    # Pre-process the case by merging cells with a Temperature less than T_threshold
    for i in range(max_iter):
        print(f"\r{(initial_number - (cluster_map.get_cluster_number() - n_clusters + 1)) / initial_number * 100:>5.2f}   "
              f"{cluster_map.get_cluster_number():<24} (Pre-processing phase - creating clusters with T < T_treshold)", end="")
        if cluster_map.merge_under_threshold():
            break
        if i == max_iter - 1:
            raise RuntimeError(f"Could not finish the pre-clustering algorithm within {max_iter} iterations, try changing"
                               f"max_iter in the function cluster_data_linkage() inside Clustering.py or search for the"
                               f"bug that prevents the algorithm from converging.")
    # Actual clustering
    for i in range(max_iter):
        print(f"\r{(initial_number - (cluster_map.get_cluster_number() - n_clusters + 1)) / initial_number * 100:>5.2f}   "
              f"{cluster_map.get_cluster_number():<24}                                                               ", end="")
        cluster_map.find_and_merge()
        if cluster_map.get_cluster_number() <= n_clusters:
            break
        if i == max_iter - 1:
            raise RuntimeError(f"Could not finish the clustering algorithm within {max_iter} iterations, try changing"
                               f"max_iter in the function cluster_data_linkage() inside Clustering.py or search for the"
                               f"bug that prevents the algorithm from converging.")
    print("")
    # "Cut" all the cluster that are too long (i.e. with cluster_length > max_ratio * grid_length) - !!! OBSOLETE !!!
    if to_cut:
        cluster_map.cut(y, z)
    if rework:
        cluster_map.rework(Ny, Nz)
    print(f"Total time for clustering: {time() - start_time:.2f} s")

    # Create the array of cluster ids and assign to each cell a unique id from 0 to n_clusters
    counter = 0
    cluster_id = np.empty((Ny, Nz), dtype=int)
    print(f"Clusters data:\n"
          f"     n°    "
          f"  T range  "
          f"   T mean  "
          f"   max T   "
          f"   min T   ")
    for cluster in cluster_map.clusters:
        if cluster.empty:
            continue
        for cell in cluster.cells:
            cluster_id[cell[0], cell[1]] = counter
        print(f"{counter:^11d}"
              f"{cluster.s_range:^11.2f}"
              f"{cluster.s_mean:^11.2f}"
              f"{cluster.max_s:^11.2f}"
              f"{cluster.min_s:^11.2f}")
        counter += 1

    # Save the results in the cache
    if save_cache:
        if not os.path.isdir(os.path.join(f'{os.getcwd()}', f'data', f'{case_name}', f'cache', f'cluster_ids')):
            os.mkdir(os.path.join(f'{os.getcwd()}', f'data', f'{case_name}', f'cache', f'cluster_ids'))
        print(f"Saving cluster_id as {os.path.join(f'{os.getcwd()}', f'data', f'{case_name}', f'cache', f'cluster_ids', f'cluster_id_{int(np.max(cluster_id) + 1)}cl.npy')} ...")
        np.save(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"cache", f"cluster_ids", f"cluster_id_{int(np.max(cluster_id) + 1)}cl.npy"), cluster_id)

    print(f"Total cluster number: {int(np.max(cluster_id) + 1)}")
    return cluster_id


def get_cluster_neighbors(Ny, Nz, cluster_id):
    neighboring_clusters = []
    for i in range(Ny):
        for j in range(Nz):
            new_neighbors = []
            if i > 0 and cluster_id[i, j] != cluster_id[i - 1, j]:
                new_neighbors.append((cluster_id[i, j], cluster_id[i - 1, j]))
            if i < Ny - 1 and cluster_id[i, j] != cluster_id[i + 1, j]:
                new_neighbors.append((cluster_id[i, j], cluster_id[i + 1, j]))
            if j > 0 and cluster_id[i, j] != cluster_id[i, j - 1]:
                new_neighbors.append((cluster_id[i, j], cluster_id[i, j - 1]))
            if j < Nz - 1 and cluster_id[i, j] != cluster_id[i, j + 1]:
                new_neighbors.append((cluster_id[i, j], cluster_id[i, j + 1]))
            for n in new_neighbors:
                if n not in neighboring_clusters:
                    neighboring_clusters.append(n)
    return neighboring_clusters


def main():
    load_cache = True
    save_cache = False

    # Load data
    case_name = "Finale"
    Ny, Nz, y, z, V, vx, vy, vz, T, rho = initialize(case_name)

    # Read input file
    _, _, _, _, _, _, _, _, T_threshold = parse_input(os.path.join(f"{os.getcwd()}", f"data", f"{case_name}", f"CRNB_input.dic"))

    # Get eddy data
    # Give id 0 to cells not in an eddy, 1 to cells in the firsts eddy, 2 to the cells in the second eddy, etc...
    eddy_id = detect_eddy(case_name, Ny, Nz, y, z, vy, vz, load_cached=True)

    n_clusters = 19
    cluster_id_linkage = cluster_data_linkage(case_name, Ny, Nz, y, z, T, n_clusters, eddy_id, T_threshold, load_cached=load_cache, save_cache=save_cache, to_cut=False, rework=False)

    # Plot of raw data and clusters
    print("Plotting results...")

    # I know that defining functions and importing libraries in the middle of the main function is something that should never happen but whatever it works and I need to graduate in two days sooooooo here it goes
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # The following two functions have been obviously copy-pasted directly from StackOverflow
    def colorbar_index(ncolors, cmap):
        cmap = cmap_discretize(cmap, ncolors)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors + 0.5)
        colorbar = plt.colorbar(mappable)
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(range(ncolors))

    def cmap_discretize(cmap, N):
        """Return a discrete colormap from the continuous colormap cmap.

            cmap: colormap instance, eg. cm.jet.
            N: number of colors.

        Example
            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
            imshow(x, cmap=djet)
        """

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
        colors_rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N + 1)
        cdict = {}
        for ki, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                          for i in range(N + 1)]
        # Return colormap object.
        return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

    f = 0.8
    fig = plt.figure(figsize=(8.3 / 1.5 * f, 8.3 * f))

    ax = fig.add_subplot(121, aspect='equal')
    ax.pcolormesh(y, z, cluster_id_linkage, cmap=plt.cm.get_cmap('tab20c', n_clusters), shading='auto')
    ax.set_title('Cluster ID')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.xaxis.set_ticks([0, y[int(y.shape[0] / 3 * 2), int(y.shape[1] * 0.6)]])
    colorbar_index(ncolors=n_clusters, cmap=plt.cm.get_cmap('tab20c', n_clusters))

    ax = fig.add_subplot(122, aspect='equal')
    cs = ax.pcolormesh(y, z, T, cmap='jet', shading='auto')
    ax.set_title('Temperature')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.xaxis.set_ticks([0, y[-1, -1]])
    fig.colorbar(cs)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
