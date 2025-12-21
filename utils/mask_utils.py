import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from scipy.ndimage import zoom, label, binary_dilation
from skimage.measure import marching_cubes
from sklearn.neighbors import NearestNeighbors
from utils.config import ROOT_PATH, MONUMENT_CONFIG, PART_COLORS, PART_COLORS_NP, INTERIOR_PARTS
# =========================
# Loading & preprocessing
# =========================

def load_mask(
    root_path, monument_name, view_name, max_dim=None
):
    import os, cv2, numpy as np

    mask_dir = os.path.join(root_path, monument_name, "masks")
    path = os.path.join(mask_dir, f"{monument_name}_{view_name}_mask.png")

    mask = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    if max_dim is not None:
        h, w = mask.shape[:2]
        s = max_dim / max(h, w)
        mask = cv2.resize(
            mask,
            (int(w * s), int(h * s)),
            interpolation=cv2.INTER_NEAREST
        )

    return mask

def load_and_prepare_masks(
    root_path, monument_name, view_name,
    max_dim, part_colors_np, interior_parts,
    visualize=False
):
    import os, cv2, numpy as np, matplotlib.pyplot as plt

    # ---- load base semantic mask (LOGIC SPACE) ----
    mask_dir = os.path.join(root_path, monument_name, "masks")
    base_path = os.path.join(mask_dir, f"{monument_name}_{view_name}_mask.png")
    semantic_mask = cv2.cvtColor(cv2.imread(base_path), cv2.COLOR_BGR2RGB)

    # ---- interior → exterior (LOGIC SPACE) ----
    interior_mask = np.any(
        [np.all(semantic_mask == part_colors_np[p], axis=-1)
         for p in interior_parts],
        axis=0
    )
    semantic_mask_exterior = semantic_mask.copy()
    semantic_mask_exterior[interior_mask] = part_colors_np["full_building"]

    # ---- resize (EXACT original behavior) ----
    def resize_to_max(img):
        h, w = img.shape[:2]
        s = max_dim / max(h, w)
        return cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_NEAREST)

    semantic_mask_resized = resize_to_max(semantic_mask)
    semantic_mask_exterior_resized = resize_to_max(semantic_mask_exterior)

    # ---- Charminar visualization-only override ----
    if monument_name == "Charminar":
        win_path = os.path.join(mask_dir, f"{monument_name}_{view_name}_mask_win.png")
        if os.path.exists(win_path):
            semantic_mask_resized = resize_to_max(
                cv2.cvtColor(cv2.imread(win_path), cv2.COLOR_BGR2RGB)
            )

    # ---- binary mask (FOR CARVING) ----
    binary_mask = (~np.all(
        semantic_mask_exterior_resized == part_colors_np["background"], axis=-1
    )).astype(np.uint8)

    # ---- visualize ----
    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(semantic_mask_resized);          axs[0].set_title("Original Mask")
        axs[1].imshow(semantic_mask_exterior_resized); axs[1].set_title("Exterior Mask")
        axs[2].imshow(binary_mask, cmap="gray");       axs[2].set_title("Binary Mask")
        for ax in axs: ax.axis("off")
        plt.tight_layout(); plt.show()

    return semantic_mask_resized, semantic_mask_exterior_resized, binary_mask

def mask_parts_from_image(image, part_colors, selected_parts):
    mask = np.zeros_like(image)

    for part in selected_parts:
        color = part_colors[part]
        match = np.all(image == color, axis=-1)
        mask[match] = color

    return mask

# def extract_top_k_components(voxel_grid, color, k=4):
#     mask = np.all(voxel_grid == color, axis=-1)
#     labeled, _ = label(mask, structure=np.ones((3, 3, 3)))
#     heights = [(i, np.ptp(np.argwhere(labeled == i)[:, 1]))
#                for i in range(1, labeled.max() + 1)]
#     top_ids = [idx for idx, _ in sorted(heights, key=lambda x: -x[1])[:k]]
#     top_mask = np.isin(labeled, top_ids)
#     filtered = voxel_grid.copy()
#     filtered[mask & (~top_mask)] = 0
#     return filtered


# =========================
# Voxel → point & plotting
# =========================

# def voxel_grid_to_points(grid, axis='z', colormap='viridis', stride=2):
#     W, H, D = grid.shape[:3]
#     is_color = grid.ndim == 4 and grid.shape[3] == 3
#     mask = np.any(grid, axis=-1) if is_color else grid != 0

#     mask_ds = mask[::stride, ::stride, ::stride]
#     zs, ys, xs = np.where(mask_ds)
#     pts = np.stack([xs, ys, zs], axis=1).astype(np.float32) * stride
#     # np.where returns (z, y, x) → convert to (x, y, z)

#     if is_color:
#         colors = grid[::stride, ::stride, ::stride][zs, ys, xs]
#     else:
#         vals = {'x': xs, 'y': ys, 'z': zs}[axis] / {'x': W - 1, 'y': H - 1, 'z': D - 1}[axis]
#         colors = (plt.get_cmap(colormap)(vals)[:, :3] * 255).astype(np.uint8)

#     return pts, colors, (H, W, D)

# def meshify_colored_voxel_grid(colored_voxel_grid, stride=1):
#     """
#     Convert a colored voxel grid (W, H, D, 3) into a surface mesh
#     with vertex colors using marching cubes and nearest-voxel coloring.

#     Parameters
#     ----------
#     colored_voxel_grid : np.ndarray
#         (W, H, D, 3)
#     stride : int
#         Downsampling factor before meshing (>=1)

#     Returns
#     -------
#     verts : (N, 3) float
#     faces : (M, 3) int
#     vertex_colors : (N, 3) float in [0,1]
#     """

#     if stride > 1:
#         grid = colored_voxel_grid[::stride, ::stride, ::stride]
#     else:
#         grid = colored_voxel_grid

#     # --- Occupancy ---
#     voxel_mask = np.any(grid > 0, axis=-1)

#     # --- Marching cubes ---
#     verts, faces, _, _ = marching_cubes(
#         voxel_mask.astype(np.uint8),
#         level=0.5
#     )

#     # Compensate for stride in vertex coordinates
#     verts = verts * stride

#     # --- Nearest voxel color assignment ---
#     filled_coords = np.argwhere(voxel_mask)
#     filled_colors = grid[voxel_mask]

#     nbrs = NearestNeighbors(n_neighbors=1).fit(filled_coords)
#     _, idx = nbrs.kneighbors(verts / stride)  # NN in downsampled space
#     vertex_colors = filled_colors[idx[:, 0]]

#     if vertex_colors.max() > 1:
#         vertex_colors = vertex_colors / 255.0

#     return verts, faces, vertex_colors



# def plot_voxel(points, colors=None):
#     if colors is None:
#         color_input = 'blue'
#     else:
#         colors = np.asarray(colors)
#         if colors.ndim == 2 and colors.shape[1] == 3 and colors.max() > 1:
#             colors = colors / 255
#         color_input = colors

#     fig = go.Figure(go.Scatter3d(
#         x=points[:, 0],
#         y=points[:, 1],
#         z=points[:, 2],
#         mode='markers',
#         marker=dict(size=2, color=color_input, opacity=1)
#     ))
#     fig.update_layout(scene=dict(aspectmode='data'), title='3D Visualization')
#     fig.show()


# def visualize_trimesh(colored):
#     occ = np.any(colored > 0, axis=-1)
#     verts, faces, normals, _ = marching_cubes(occ.astype(float), 0.5)
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
#     mesh.show()

# def visualize_mesh_plotly(verts, faces, vertex_colors=None, title="Colored Voxel Mesh"):
#     """
#     Visualize a mesh with optional vertex colors using Plotly.
#     """

#     if vertex_colors is not None and vertex_colors.max() > 1:
#         vertex_colors = vertex_colors / 255.0

#     fig = go.Figure(go.Mesh3d(
#         x=verts[:, 0],
#         y=verts[:, 1],
#         z=verts[:, 2],
#         i=faces[:, 0],
#         j=faces[:, 1],
#         k=faces[:, 2],
#         vertexcolor=vertex_colors,
#         flatshading=True,
#         opacity=1,
#         lighting=dict(ambient=1),
#     ))

#     fig.update_layout(
#         scene=dict(aspectmode="data"),
#         title=title
#     )
#     fig.show()
