# utils/eval_helpers.py

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from utils.preprocess_helpers import normalize_preserve_aspect


# ============================================================
# Geometry helpers
# ============================================================

def filter_mesh(vertices, faces, y_thresh=0.2):
    mask = vertices[:, 1] <= y_thresh
    valid_idx = np.where(mask)[0]
    face_mask = np.all(np.isin(faces, valid_idx), axis=1)
    return vertices[mask], faces[face_mask]


# ============================================================
# Accuracy metrics
# ============================================================

def _downsample(P, n=20000):
    if len(P) <= n:
        return P
    idx = np.random.choice(len(P), n, replace=False)
    return P[idx]


def chamfer_distance(A, B, max_points=20000, squared=True):
    A = _downsample(A, max_points)
    B = _downsample(B, max_points)

    treeB = cKDTree(B)
    dA, _ = treeB.query(A, k=1, workers=-1)

    treeA = cKDTree(A)
    dB, _ = treeA.query(B, k=1, workers=-1)

    if squared:
        return float(np.mean(dA**2) + np.mean(dB**2))
    else:
        return float(np.mean(dA) + np.mean(dB))


def fscore_with_threshold(A, B, tau=0.03, max_points=20000):
    A = _downsample(A, max_points)
    B = _downsample(B, max_points)

    nn_AB = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(B)
    d_AB, _ = nn_AB.kneighbors(A)
    precision = float(np.mean(d_AB[:, 0] < tau))

    nn_BA = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(A)
    d_BA, _ = nn_BA.kneighbors(B)
    recall = float(np.mean(d_BA[:, 0] < tau))

    f1 = 0.0 if (precision + recall) == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return f1, precision, recall


def pca_shape_similarity(A, B):
    pca_A = PCA(n_components=3).fit(A)
    pca_B = PCA(n_components=3).fit(B)
    return 1.0 - np.sum(
        np.abs(pca_A.explained_variance_ratio_ -
               pca_B.explained_variance_ratio_)
    )


# ============================================================
# Completeness metrics
# ============================================================

def voxel_iou(A, B, resolution=96, dilate_frac=0.01):
    all_pts = np.vstack([A, B])
    bounds_min, bounds_max = all_pts.min(0), all_pts.max(0)
    step = (bounds_max - bounds_min).max() / resolution

    def to_occ(points):
        idx = ((points - bounds_min) / step).astype(int)
        idx = np.clip(idx, 0, resolution - 1)
        occ = np.zeros((resolution,) * 3, dtype=bool)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return occ

    occA = to_occ(A)
    occB = to_occ(B)

    if dilate_frac > 0:
        iters = max(1, int(round(
            (dilate_frac * np.linalg.norm(bounds_max - bounds_min)) / step
        )))
        occA = binary_dilation(occA, iterations=iters)
        occB = binary_dilation(occB, iterations=iters)

    inter = np.count_nonzero(occA & occB)
    union = np.count_nonzero(occA | occB)
    return inter / union if union > 0 else np.nan


# ============================================================
# Regularity metrics
# ============================================================

def compute_nn_stats(pts, max_points=50000):
    if len(pts) > max_points:
        pts = pts[np.random.choice(len(pts), max_points, replace=False)]

    nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(pts)
    distances, _ = nbrs.kneighbors(pts)

    nn = distances[:, 1]
    return {
        "NN Mean ↓": nn.mean(),
        "NN Std ↓": nn.std(),
        "NN CV ↓": nn.std() / (nn.mean() + 1e-8)
    }


# ============================================================
# Surface + marching cubes
# ============================================================

# def get_marching_cubes_mesh(points, grid_size=128, sigma=1.0, level=0.1):
#     mins = points.min(0)
#     maxs = points.max(0)
#     span = np.maximum(maxs - mins, 1e-8)
#     p = (points - mins) / span

#     vox = np.clip(
#         (p * (grid_size - 1)).round().astype(np.int32),
#         0, grid_size - 1
#     )

#     grid = np.zeros((grid_size,) * 3, dtype=np.float32)
#     np.add.at(grid, (vox[:, 0], vox[:, 1], vox[:, 2]), 1.0)

#     if sigma > 0:
#         grid = gaussian_filter(grid, sigma=sigma)

#     verts, faces, _, _ = marching_cubes(grid, level=level)
#     verts = verts / (grid_size - 1) * span + mins
#     return verts, faces

# def get_marching_cubes_mesh(points, grid_size=128, sigma=1.0, level=0.1):
#     # normalize exactly like notebook
#     min_val = points.min(0)
#     size = points.max(0) - min_val
#     scale = size.max()

#     p = (points - min_val) / (scale + 1e-8)
#     vox = np.clip(
#         (p * (grid_size - 1)).round().astype(np.int32),
#         0, grid_size - 1
#     )

#     grid = np.zeros((grid_size,) * 3, dtype=np.float32)
#     np.add.at(grid, (vox[:, 0], vox[:, 1], vox[:, 2]), 1.0)

#     if sigma > 0:
#         grid = gaussian_filter(grid, sigma=sigma)

#     verts, faces, _, _ = marching_cubes(grid, level=level)

#     # back to normalized unit cube
#     verts /= grid_size
#     return verts, faces

def pointcloud_to_voxel_grid(points, grid_size=128, sigma=1.0):
    norm_points = normalize_preserve_aspect(points)
    voxel_coords = (norm_points * (grid_size - 1)).astype(int)
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    np.add.at(grid, (voxel_coords[:,0], voxel_coords[:,1], voxel_coords[:,2]), 1)
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    # Clamp boundary
    grid[[0, -1], :, :] = 0
    grid[:, [0, -1], :] = 0
    grid[:, :, [0, -1]] = 0
    return grid
    
def get_marching_cubes_mesh(points, grid_size=128, sigma=1.0, level=0.1):
    voxel_grid = pointcloud_to_voxel_grid(points, grid_size, sigma)
    verts, faces, _, _ = marching_cubes(voxel_grid, level=level)
    verts /= grid_size
    return verts, faces


def compute_triangle_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)


def compute_vertex_normals(vertices, faces):
    tri_normals = compute_triangle_normals(vertices, faces)
    vnorm = np.zeros_like(vertices)
    for i in range(faces.shape[0]):
        for j in range(3):
            vnorm[faces[i, j]] += tri_normals[i]
    return vnorm / (np.linalg.norm(vnorm, axis=1, keepdims=True) + 1e-8)


def compute_surface_metrics(vertices, faces, k=20):
    normals = compute_vertex_normals(vertices, faces)
    nbrs = NearestNeighbors(n_neighbors=k).fit(vertices)
    _, indices = nbrs.kneighbors(vertices)

    normal_stds = []
    roughness_vals = []
    mean_curvatures = []

    for i, nbr_idx in enumerate(indices):
        nbr_pts = vertices[nbr_idx]
        center = vertices[i]
        center_normal = normals[i]

        nbr_normals = normals[nbr_idx]
        dot = np.clip(nbr_normals @ center_normal, -1.0, 1.0)
        angles = np.degrees(np.arccos(dot))
        normal_stds.append(np.std(angles))

        pca = PCA(n_components=3).fit(nbr_pts)
        roughness_vals.append(pca.explained_variance_[2])

        laplace = nbr_pts.mean(axis=0) - center
        mean_curvatures.append(np.linalg.norm(laplace))

    return {
        "Normal StdDev (°)": np.mean(normal_stds),
        "Mean Roughness (λ₃)": np.mean(roughness_vals),
        "Mean Curvature": np.mean(mean_curvatures),
    }

__all__ = [
    "filter_mesh",
    "chamfer_distance",
    "fscore_with_threshold",
    "pca_shape_similarity",
    "voxel_iou",
    "compute_nn_stats",
    "get_marching_cubes_mesh",
    "compute_surface_metrics",
]
