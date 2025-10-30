import numpy as np
import trimesh
import scipy.sparse.linalg
import robust_laplacian

def compute_hks(L, M, num_eigen=500, T=100):
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(L, M=M, k=num_eigen, which='SM')
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    t_min = 4.0 / eigvals[-1]
    t_max = 4.0 / eigvals[1]
    time_scales = np.logspace(np.log10(t_min), np.log10(t_max), T)

    hks_all_times = []
    for t in time_scales:
        exp_eig = np.exp(-t * eigvals)
        hks_t = (eigvecs ** 2) @ exp_eig
        hks_all_times.append(hks_t)

    hks = np.array(hks_all_times)
    hks /= hks.max(axis=1, keepdims=True)
    return hks, time_scales

hks, time_scales = compute_hks(L, M, num_eigen=100, T=100)

def find_keypoints(hks, mesh, threshold=0.9):
    adjacency = mesh.vertex_neighbors
    num_scales, num_vertices = hks.shape
    keypoints_idx = set()
    scales_to_use = np.linspace(0, num_scales - 1, 5, dtype=int)

    for s in scales_to_use:
        hks_t = hks[s]
        for v in range(num_vertices):
            if all(hks_t[v] > hks_t[n] for n in adjacency[v]):
                keypoints_idx.add(v)

    hks_global = hks.max(axis=0)
    values = np.array([hks_global[i] for i in keypoints_idx])
    cutoff = np.percentile(values, threshold * 100)
    return [i for i in keypoints_idx if hks_global[i] >= cutoff]
