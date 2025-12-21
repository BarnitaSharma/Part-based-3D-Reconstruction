import numpy as np
import scipy.ndimage
from scipy.ndimage import label
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage.measure import marching_cubes
import trimesh
from sklearn.neighbors import NearestNeighbors

from utils.config import PART_COLORS, PART_COLORS_NP
from utils.voxel_utils import *
from utils.visualization import *
# =========================================================
# Internal helpers (PRIVATE)
# =========================================================

def _mask_to_wh(mask, W, H):
    """
    Ensure mask is (W, H).
    Accepts (H, W) or (W, H).
    """
    if mask.shape[:2] == (H, W):
        return mask.T
    if mask.shape[:2] == (W, H):
        return mask
    raise ValueError(f"Mask shape {mask.shape} incompatible with (W,H)=({W},{H})")



def _occupancy(grid):
    return np.any(grid > 0, axis=-1).astype(np.uint8)


def _project_mask_2d_to_3d(mask2d, depth):
    return np.repeat(mask2d[:, :, None], depth, axis=2)


def _bbox_from_mask(mask3d):
    coords = np.argwhere(mask3d)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return (*lo, *hi)


def _rotate_and_carve(grid, mask, angle_interval):
    center = np.array(grid.shape) / 2
    out = grid
    for angle in range(0, 91, angle_interval):
        out = scipy.ndimage.affine_transform(
            out,
            _rotation_matrix_inv(angle),
            offset=center - _rotation_matrix_inv(angle) @ center,
            order=1,
            mode="constant",
            cval=0,
        )
        out = carve_voxel_grid_with_masks(out, mask)
    return out


def _rotation_matrix_inv(angle):
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.linalg.inv(R)


# =========================================================
# Public API (UNCHANGED)
# =========================================================

def carve_voxel_grid_with_masks(voxel_grid, combined_mask):
    is_color = voxel_grid.ndim == 4
    W, H, D = voxel_grid.shape[:3]

    mask = _mask_to_wh(combined_mask, W, H)

    # Binary mask
    if mask.ndim == 2:
        m3 = mask[:, :, None]           # (W, H, 1)
        if is_color:
            m3 = m3[:, :, :, None]      # (W, H, 1, 1)
        return np.where(m3, voxel_grid, 0)

    # RGB mask
    if mask.ndim == 3 and mask.shape[2] == 3:
        out = voxel_grid.copy()
        for c in range(3):
            mc = mask[:, :, c][:, :, None, None]
            out[..., c] = np.where(mc, voxel_grid[..., c], 0)
        return out

    raise ValueError("Unsupported mask shape")



# def process_voxel_grid(voxel_grid, combined_mask, angle_interval=90):
#     return _rotate_and_carve(voxel_grid, combined_mask, angle_interval)

def process_voxel_grid(voxel_grid, combined_mask, angle_interval=90):
    """
    Rotate + carve with visible progress.
    """
    rotation_center = np.array(voxel_grid.shape) / 2
    carved_grid = voxel_grid

    for angle in tqdm(
        range(0, 91, angle_interval),
        desc="90 Carving",
        leave=True
    ):
        carved_grid = scipy.ndimage.affine_transform(
            carved_grid,
            _rotation_matrix_inv(angle),
            offset=rotation_center - _rotation_matrix_inv(angle) @ rotation_center,
            order=1,
            mode="constant",
            cval=0,
        )
        carved_grid = carve_voxel_grid_with_masks(carved_grid, combined_mask)

    return carved_grid

def apply_colored_mask_to_voxel_grid(carved_voxel_grid, colored_mask):
    W, H, D = carved_voxel_grid.shape
    mask = colored_mask.transpose(1, 0, 2)
    mask3 = np.repeat(mask[:, :, None, :], D, axis=2)

    out = np.zeros((W, H, D, 3), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.where(carved_voxel_grid == 1, mask3[..., c], 0)
    return out


def part_carve(colored_grid, semantic_mask, group_jobs, visualize=False):
    final = np.zeros_like(colored_grid)

    for names, angle in group_jobs:
        mask2d = np.any(
            [np.all(semantic_mask == PART_COLORS[n], axis=-1) for n in names],
            axis=0,
        )

        if not mask2d.any():
            continue

        m = mask2d.T.astype(np.uint8)
        sub = colored_grid * m[:, :, None, None]

        occ = _occupancy(sub)
        carved = process_voxel_grid(occ, m, angle)
        part = sub * carved[:, :, :, None]

        final[np.any(part > 0, axis=-1)] = part[np.any(part > 0, axis=-1)]

    return final


def left_right_guided_carve(colored_grid, semantic_mask, target_color, angle=60, visualize=False, stride = 2):
    """
    Component-guided carving with minimal logs.
    """
    W, H, D, _ = colored_grid.shape
    carved_grid = colored_grid.copy()

    mask2d = np.all(semantic_mask == target_color, axis=-1)
    if not np.any(mask2d):
        print(f"[SKIP] No mask for color {target_color}")
        return carved_grid

    labeled_3d, num_3d = label(np.all(colored_grid == target_color, axis=-1))
    print(f"[{target_color}] 3D components: {num_3d}")

    for i in range(1, num_3d + 1):
        mask3d = labeled_3d == i
        coords = np.argwhere(mask3d)
        if coords.size == 0:
            continue

        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1

        print(f"  - Component {i}: bbox ({x0},{y0},{z0}) → ({x1},{y1},{z1})")

        crop2d = mask2d[y0:y1, x0:x1]
        subgrid = colored_grid[x0:x1, y0:y1, z0:z1].copy()

        occ = np.any(subgrid > 0, axis=-1).astype(np.uint8)
        carved_occ = process_voxel_grid(occ, crop2d, angle_interval=angle)

        print(f"    carved voxels: {np.count_nonzero(carved_occ)}")

        carved_color = subgrid * carved_occ[:, :, :, None]

        carved_grid[x0:x1, y0:y1, z0:z1][mask3d[x0:x1, y0:y1, z0:z1]] = 0
        mask_any = np.any(carved_color > 0, axis=-1)
        carved_grid[x0:x1, y0:y1, z0:z1][mask_any] = carved_color[mask_any]

        if visualize:
            pts, cols, _ = voxel_grid_to_points(carved_color, stride=stride)
            if pts.shape[0] > 0:
                plot_voxel(pts, cols)



    return carved_grid


def extrude_from_surface(grid, mask_2d, axis, direction="+", depth=5, fill_color=None):
    occ = _occupancy(grid)
    W, H, D = occ.shape
    filled = np.zeros_like(occ, bool)

    if axis == 2:
        start = np.argmax(occ if direction == "+" else occ[:, :, ::-1], axis=2)
        if direction == "-":
            start = D - 1 - start
        valid = mask_2d.T

        for d in range(depth):
            z = start + d if direction == "+" else start - d
            ok = (z >= 0) & (z < D) & valid
            xs, ys = np.nonzero(ok)
            filled[xs, ys, z[xs, ys]] = True

    elif axis == 0:
        start = np.argmax(occ if direction == "+" else occ[::-1], axis=0)
        if direction == "-":
            start = W - 1 - start
        valid = mask_2d

        for d in range(depth):
            x = start + d if direction == "+" else start - d
            ok = (x >= 0) & (x < W) & valid
            ys, zs = np.nonzero(ok)
            filled[x[ys, zs], ys, zs] = True

    out = grid.copy()
    if fill_color is None:
        out[filled] = 0
    else:
        out[filled] = fill_color

    return out



def recolor_backward_components(voxel_grid, color, new_color, k=4, sort_axis=2):
    mask = np.all(voxel_grid == color, axis=-1)
    labeled, n = label(mask)

    comps = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        comps.append((i, coords[:, sort_axis].mean()))

    keep = {i for i, _ in sorted(comps, key=lambda x: x[1])[:k]}
    out = voxel_grid.copy()
    for i in range(1, n + 1):
        if i not in keep:
            out[labeled == i] = new_color
    return out


def global_carve(
    binary_mask,
    semantic_mask_exterior,
    angle_interval=90,
    stride=4,
    visualize=False
):

    # Build initial grid
    h, w = binary_mask.shape
    voxel_grid = np.ones((w, h, w), dtype=np.uint8)

    # Binary carving
    carved_binary = process_voxel_grid(
        voxel_grid,
        binary_mask,
        angle_interval=angle_interval,
    )

    # Apply semantic colors
    colored_voxel = apply_colored_mask_to_voxel_grid(
        carved_binary,
        semantic_mask_exterior,
    )
    if visualize:
        pts, cols, _ = voxel_grid_to_points(colored_voxel, stride=stride)
        if pts.shape[0] > 0:
            plot_voxel(pts, cols, title="After global symmetric carving")

    return colored_voxel



def partwise_carve(
    colored_voxel_grid,
    semantic_mask_exterior,
    semantic_mask_full,
    part_colors_np,
    group_jobs,
    part_symmetry,
    extrusion_depths,
    recolor_back_minarets=True,
    visualize=False,
    stride=4
):
    """
    Part-wise refinement after global carving.
    """

    grid = colored_voxel_grid

    # -------------------------------------------------
    # 1. Global symmetry per part group
    # -------------------------------------------------
    grid = part_carve(
        grid,
        semantic_mask_exterior,
        group_jobs,
        visualize=False,   # visualization handled here
    )

    if visualize:
        pts, cols, _ = voxel_grid_to_points(grid, stride=stride)
        if pts.shape[0] > 0:
            plot_voxel(pts, cols, title="After part-wise symmetric carving (global symmetry on each part)")

    # -------------------------------------------------
    # 2. Component-guided symmetry
    # -------------------------------------------------
    for part, angle in part_symmetry.items():
        grid = left_right_guided_carve(
            colored_grid=grid,
            semantic_mask=semantic_mask_exterior,
            target_color=part_colors_np[part],
            angle=angle,
            visualize=False,
            stride=stride,
        )

    if visualize:
        pts, cols, _ = voxel_grid_to_points(grid, stride=stride)
        if pts.shape[0] > 0:
            plot_voxel(pts, cols, title="After part-wise symmetric carving (local symmetry on each part)")

    # -------------------------------------------------
    # 3. Interior extrusion (doors / windows)
    # -------------------------------------------------
    def extrude_4dirs(grid, mask, depth, color):
        grid = extrude_from_surface(grid, mask, axis=2, direction="+", depth=depth, fill_color=color)
        grid = extrude_from_surface(grid, mask, axis=2, direction="-", depth=depth, fill_color=color)
        grid = extrude_from_surface(grid, mask, axis=0, direction="+", depth=depth, fill_color=color)
        grid = extrude_from_surface(grid, mask, axis=0, direction="-", depth=depth, fill_color=color)
        return grid

    for part, depth in extrusion_depths.items():
        mask = np.all(
            semantic_mask_full == part_colors_np[part],
            axis=-1,
        )
        grid = extrude_4dirs(
            grid,
            mask,
            depth,
            part_colors_np[part],
        )

    if visualize:
        pts, cols, _ = voxel_grid_to_points(grid, stride=stride)
        if pts.shape[0] > 0:
            plot_voxel(pts, cols, title="After interior extrusion")

    # -------------------------------------------------
    # 4. Recolor back minarets
    # -------------------------------------------------
    if recolor_back_minarets:
        oriented = grid.transpose(2, 1, 0, 3)
        oriented = np.flip(oriented, axis=1)

        grid = recolor_backward_components(
            oriented,
            part_colors_np["front_minarets"],
            new_color=part_colors_np["back_minarets"],
            k=2,
            sort_axis=0,
        )

        if visualize:
            pts, cols, _ = voxel_grid_to_points(grid, stride=stride)
            if pts.shape[0] > 0:
                plot_voxel(pts, cols, title="After back-minaret recoloring")

    return grid


