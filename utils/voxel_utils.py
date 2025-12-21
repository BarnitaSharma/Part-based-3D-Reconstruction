import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors

def get_voxel_points_by_parts(grid, part_colors, part_names):
    """
    part_colors[name] == (R,G,B)
    """
    mask = np.zeros(grid.shape[:3], dtype=bool)

    for name in part_names:
        color = part_colors[name]
        mask |= np.all(grid == color, axis=-1)

    z, y, x = np.where(mask)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = grid[z, y, x]

    return pts, colors


def extract_top_k_components(voxel_grid, color, k=4):
    mask = np.all(voxel_grid == color, axis=-1)
    labeled, _ = label(mask, structure=np.ones((3, 3, 3)))
    heights = [(i, np.ptp(np.argwhere(labeled == i)[:, 1]))
               for i in range(1, labeled.max() + 1)]
    top_ids = [idx for idx, _ in sorted(heights, key=lambda x: -x[1])[:k]]
    top_mask = np.isin(labeled, top_ids)
    filtered = voxel_grid.copy()
    filtered[mask & (~top_mask)] = 0
    return filtered

def voxel_grid_to_points(grid, axis='z', colormap='viridis', stride=2):
    W, H, D = grid.shape[:3]
    is_color = grid.ndim == 4 and grid.shape[3] == 3
    mask = np.any(grid, axis=-1) if is_color else grid != 0

    mask_ds = mask[::stride, ::stride, ::stride]
    zs, ys, xs = np.where(mask_ds)
    pts = np.stack([xs, ys, zs], axis=1).astype(np.float32) * stride
    # np.where returns (z, y, x) → convert to (x, y, z)

    if is_color:
        colors = grid[::stride, ::stride, ::stride][zs, ys, xs]
    else:
        vals = {'x': xs, 'y': ys, 'z': zs}[axis] / {'x': W - 1, 'y': H - 1, 'z': D - 1}[axis]
        colors = (plt.get_cmap(colormap)(vals)[:, :3] * 255).astype(np.uint8)

    return pts, colors, (H, W, D)

def meshify_colored_voxel_grid(colored_voxel_grid, stride=1):
    """
    Convert a colored voxel grid (W, H, D, 3) into a surface mesh
    with vertex colors using marching cubes and nearest-voxel coloring.
    """

    # --- Downsample if needed ---
    if stride > 1:
        grid = colored_voxel_grid[::stride, ::stride, ::stride]
    else:
        grid = colored_voxel_grid

    # --- Occupancy ---
    voxel_mask = np.any(grid > 0, axis=-1)

    # --- Marching cubes (OUTPUT IS z,y,x) ---
    verts, faces, normals, _ = marching_cubes(
        voxel_mask.astype(np.uint8),
        level=0.5
    )

    # --- Compensate stride ---
    verts = verts * stride

    # --- Fix axis order: (z,y,x) → (x,y,z) ---
    verts = verts[:, [2, 1, 0]]

    # --- Fix mirror caused by earlier transpose + flip ---
    # (front/back alignment with plot_voxel & camera space)
    verts[:, 2] = colored_voxel_grid.shape[2] - verts[:, 2]

    # --- Nearest voxel color assignment ---
    filled_coords = np.argwhere(voxel_mask)          # (z,y,x)
    filled_colors = grid[voxel_mask]

    nbrs = NearestNeighbors(n_neighbors=1).fit(filled_coords)
    _, idx = nbrs.kneighbors(verts[:, [2, 1, 0]] / stride)
    vertex_colors = filled_colors[idx[:, 0]]

    if vertex_colors.max() > 1:
        vertex_colors = vertex_colors / 255.0

    return verts, faces, vertex_colors, normals

