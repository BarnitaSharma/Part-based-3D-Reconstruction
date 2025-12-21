import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from PIL import Image
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.ndimage import label
from skimage.measure import regionprops, label as label2d
from utils.camera_geometry import *
from utils.camera_estimation import *
from utils.voxel_utils import *
from utils.mask_utils import *


# --- Data loading ---

def load_voxel_grid(npz_path):
    data = np.load(npz_path)
    voxel_grid = data["voxel_grid"]  # (X,Y,Z,3)
    # voxel_grid = np.flip(voxel_grid, axis=1)
    return voxel_grid

def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    img = Image.open(mask_path).convert("RGB")
    return np.array(img)

def resize_mask_to_voxel_grid(mask_img, voxel_grid):
    """
    Resize mask so that its max dimension matches voxel grid max dimension.
    Preserves aspect ratio. Uses nearest neighbor (label-safe).
    """
    H, W = mask_img.shape[:2]
    X, Y, Z = voxel_grid.shape[:3]

    target_max = max(X, Y, Z)
    current_max = max(H, W)

    scale = target_max / current_max

    new_W = int(round(W * scale))
    new_H = int(round(H * scale))

    resized = cv2.resize(
        mask_img,
        (new_W, new_H),
        interpolation=cv2.INTER_NEAREST
    )

    print(f"Mask resized: ({H},{W}) → ({new_H},{new_W}) | scale={scale:.3f}")
    return resized

def load_camera_json(path, view):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        data = json.load(f)

    if view not in data:
        raise KeyError(f"View '{view}' not found in {os.path.basename(path)}")

    cam = data[view]

    cam_out = {
        "cam_pos": np.array(cam["cam_pos"], dtype=np.float32),
        "target":  np.array(cam["target"],  dtype=np.float32),
        "f":  float(cam["f"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
    }
    return cam_out


def project_keypoints(voxel_kps, cam):
    return {
        k: project(pt, cam["cam_pos"], cam["target"], cam["f"], cam["cx"], cam["cy"])
        for k, pt in voxel_kps.items()
    }



# --- Visualization ---


def visualize_minaret_kp(monument, tag, cam, mask_img, voxel_kps, image_kps, minarets,back_top_only):
    proj_kps = project_keypoints(voxel_kps, cam)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_img)
    plt.title(f"{monument} | {tag} | Minaret KP reprojection")
    plt.axis("off")

    for m in minarets:
        # --- TOP always ---
        k = f"{m}_top"
        gt = image_kps[k]
        pr = proj_kps[k]

        plt.scatter(gt[0], gt[1], c="lime", s=25)
        plt.scatter(pr[0], pr[1], c="red", s=25)
        plt.plot(
            [gt[0], pr[0]],
            [gt[1], pr[1]],
            color="yellow",
            linewidth=1
        )

        # --- BOTTOM only if allowed ---
        if not (m in ["LM2", "RM2"] and back_top_only[monument]):
            k = f"{m}_bottom"
            gt = image_kps[k]
            pr = proj_kps[k]

            plt.scatter(gt[0], gt[1], c="lime", s=25)
            plt.scatter(pr[0], pr[1], c="red", s=25)
            plt.plot(
                [gt[0], pr[0]],
                [gt[1], pr[1]],
                color="yellow",
                linewidth=1
            )

    plt.show()


# ------------------------------------------------------------
# GLOBAL DEPTH BUFFER (WHOLE OBJECT)
# ------------------------------------------------------------

def compute_global_depth_buffer(voxel_grid, cam, H, W):
    R = look_at_rotation(cam["cam_pos"], cam["target"])
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    z, y, x = np.where(np.any(voxel_grid > 0, axis=-1))
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    pts_cam = (pts - cam["cam_pos"]) @ R.T
    X, Y, Z = pts_cam.T

    valid = Z > 1e-6
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = (X / Z) * cam["f"] + cam["cx"]
    v = -(Y / Z) * cam["f"] + cam["cy"]

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z = ui[inside], vi[inside], Z[inside]

    for x, y, z in zip(ui, vi, Z):
        if z < zbuf[y, x]:
            zbuf[y, x] = z

    return zbuf



# ------------------------------------------------------------
# FAST VISIBILITY-AWARE PROJECTION (PART STYLE)
# ------------------------------------------------------------

def project_part_visible(pts3d, cam, zbuf, H, W, eps=1e-3):
    R = look_at_rotation(cam["cam_pos"], cam["target"])
    pts_cam = (pts3d - cam["cam_pos"]) @ R.T
    X, Y, Z = pts_cam.T

    valid = Z > 1e-6
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = (X / Z) * cam["f"] + cam["cx"]
    v = -(Y / Z) * cam["f"] + cam["cy"]

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z = ui[inside], vi[inside], Z[inside]

    mask = np.zeros((H, W), dtype=bool)
    for x, y, z in zip(ui, vi, Z):
        if abs(z - zbuf[y, x]) < eps:
            mask[y, x] = True

    return mask


# ------------------------------------------------------------
# SIDE-BY-SIDE VISUALIZATION
# ------------------------------------------------------------

def visualize_side_by_side(gt, pr_init, pr_final, title, i0, i1):
    vis_i = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis_f = np.zeros((*gt.shape, 3), dtype=np.uint8)

    for vis, pr in [(vis_i, pr_init), (vis_f, pr_final)]:
        vis[gt] = [0, 255, 0]
        vis[pr] = [255, 0, 0]
        vis[gt & pr] = [255, 255, 0]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(vis_i)
    plt.title(f"{title} | init | IoU={i0:.3f}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(vis_f)
    plt.title(f"{title} | final | IoU={i1:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# VISUALIZATION: ALL MINARETS TOGETHER (UNCHANGED LOGIC)
# ------------------------------------------------------------

def visualize_minarets_all_3cams(voxel_init, mask_img, cams, H, W,part_colors):
    mask_min = mask_parts_from_image(
        mask_img, part_colors,
        ["front_minarets", "back_minarets"]
    )
    gt = np.any(mask_min > 0, axis=-1)

    pts, _ = get_voxel_points_by_parts(
        voxel_init, part_colors,
        ["front_minarets", "back_minarets"]
    )

    vis = {}
    ious = {}

    for tag, cam in cams.items():
        zbuf = compute_global_depth_buffer(voxel_init, cam, H, W)
        pr = project_part_visible(pts, cam, zbuf, H, W)

        ious[tag] = _iou_bool(gt, pr)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[gt] = [0, 255, 0]
        img[pr] = [255, 0, 0]
        img[gt & pr] = [255, 255, 0]
        vis[tag] = img

    plt.figure(figsize=(18, 5))
    for i, tag in enumerate(["init", "rep", "final"], 1):
        plt.subplot(1, 3, i)
        plt.imshow(vis[tag])
        plt.title(f"minarets | {tag} | IoU={ious[tag]:.3f}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return ious



# --- Evaluation ---

def _iou_bool(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else np.nan

### Binary IoU ###
def compute_binary_gt(mask_img, voxel_grid):
    # get all colors present in voxels except black
    voxel_colors = np.unique(
        voxel_grid.reshape(-1, voxel_grid.shape[-1]),
        axis=0
    )
    voxel_colors = voxel_colors[~np.all(voxel_colors == 0, axis=1)]

    gt = np.zeros(mask_img.shape[:2], dtype=bool)
    for c in voxel_colors:
        gt |= np.all(mask_img == c, axis=-1)
    return gt

def run_minaret_kp_evaluation(
    monuments,
    view,
    root_voxels,
    root_masks,
    cam_dir,
    part_colors,
    visualize=True,
):
    """
    Minaret keypoint reprojection evaluation (init → kp).
    Prints per-monument errors and final table.
    """

    minarets = ["LM1", "RM1", "LM2", "RM2"]

    back_top_only = {
        "Itimad": True,
        "Akbar": True,
        "Charminar": True,
        "Taj": False,
        "Bibi": False,
    }

    MONUMENT_SHORT = {
        "Taj": "TM",
        "Bibi": "BkM",
        "Itimad": "IuD",
        "Akbar": "AT",
        "Charminar": "CM",
    }

    # positional colors (extractor contract)
    minaret_colors = [
        part_colors["front_minarets"],
        part_colors["back_minarets"],
    ]

    cells = {m: {} for m in minarets + ["Average"]}

    for monument in monuments:
        print(f"\n🏛️ {monument}")

        voxel_grid = load_voxel_grid(
            os.path.join(root_voxels, f"{monument}_voxel_grid.npz")
        )

        mask_img = load_mask(
            os.path.join(
                root_masks, monument, "masks",
                f"{monument}_{view}_mask.png"
            )
        )
        mask_img = resize_mask_to_voxel_grid(mask_img, voxel_grid)

        cams = {
            "init": load_camera_json(
                os.path.join(cam_dir, f"{monument}_camera_params_init.json"),
                view,
            ),
            "rep": load_camera_json(
                os.path.join(cam_dir, f"{monument}_camera_params_kp.json"),
                view,
            ),
        }

        # ---- extract minarets ----
        vox_parts = extract_minaret_voxels_by_label(
            voxel_grid, minaret_colors
        )
        msk_parts = extract_minaret_masks_by_label(
            mask_img, minaret_colors
        )

        voxel_kps = extract_top_bottom_voxel_points(vox_parts)
        image_kps = extract_top_bottom_image_points(msk_parts)

        err_vals = {tag: {} for tag in cams}

        for tag, cam in cams.items():
            proj_kps = project_keypoints(voxel_kps, cam)

            if visualize:
                visualize_minaret_kp(
                    monument, tag, cam,
                    mask_img, voxel_kps, image_kps,
                    minarets, back_top_only
                )

            for m in minarets:
                errs = [
                    np.linalg.norm(
                        np.array(image_kps[f"{m}_top"]) -
                        np.array(proj_kps[f"{m}_top"])
                    )
                ]

                if not (m in ["LM2", "RM2"] and back_top_only[monument]):
                    errs.append(
                        np.linalg.norm(
                            np.array(image_kps[f"{m}_bottom"]) -
                            np.array(proj_kps[f"{m}_bottom"])
                        )
                    )

                err_vals[tag][m] = np.mean(errs)

        for m in minarets:
            cells[m][monument] = (
                f"{err_vals['init'][m]:.2f}→{err_vals['rep'][m]:.2f}"
            )

        cells["Average"][monument] = (
            f"{np.mean(list(err_vals['init'].values())):.2f}"
            f"→{np.mean(list(err_vals['rep'].values())):.2f}"
        )

    # ---- final table ----
    df = pd.DataFrame.from_dict(cells, orient="index")
    df = df[[m for m in monuments]]
    df.columns = [MONUMENT_SHORT[m] for m in df.columns]

    print(
        """
=== Minaret Keypoint Reprojection Error (px) ===
Θinit → Θkp

Rules:
- LM1, RM1: top + bottom
- LM2, RM2:
    * Taj, Bibi: top + bottom
    * Akbar, Charminar, Itimad: top only
"""
    )

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=True))

    return df


def run_minaret_iou_evaluation(
    monuments,
    view,
    root_voxels,
    root_masks,
    cam_dir,
    part_colors,
    visualize=True,
):
    """
    Minaret IoU evaluation (init → kp → final).
    Uses global visibility masking (same as original notebook).
    """

    minarets = ["LM1", "RM1", "LM2", "RM2"]

    MONUMENT_SHORT = {
        "Taj": "TM",
        "Bibi": "BkM",
        "Itimad": "IuD",
        "Akbar": "AT",
        "Charminar": "CM",
    }

    # positional colors (extractor contract)
    minaret_colors = [
        part_colors["front_minarets"],
        part_colors["back_minarets"],
    ]

    cells = {m: {} for m in minarets + ["Average"]}

    for monument in monuments:
        print(f"\n🏛️ {monument}")

        # ---- load voxel grid ----
        voxel_init = load_voxel_grid(
            os.path.join(root_voxels, f"{monument}_voxel_grid.npz")
        )

        # ---- load + resize mask ----
        mask_img = load_mask(
            os.path.join(
                root_masks, monument, "masks",
                f"{monument}_{view}_mask.png"
            )
        )
        mask_img = resize_mask_to_voxel_grid(mask_img, voxel_init)
        H, W = mask_img.shape[:2]

        # ---- load cameras ----
        cams = {
            "init": load_camera_json(
                os.path.join(cam_dir, f"{monument}_camera_params_init.json"),
                view,
            ),
            "rep": load_camera_json(
                os.path.join(cam_dir, f"{monument}_camera_params_kp.json"),
                view,
            ),
            "final": load_camera_json(
                os.path.join(cam_dir, f"{monument}_camera_params_final.json"),
                view,
            ),
        }

        # ---- VISUAL (same as notebook) ----
        if visualize:
            visualize_minarets_all_3cams(
                voxel_init, mask_img, cams, H, W, part_colors
            )

        # ---- extract minarets ----
        vox_parts = extract_minaret_voxels_by_label(
            voxel_init, minaret_colors
        )
        msk_parts = extract_minaret_masks_by_label(
            mask_img, minaret_colors
        )

        iou_vals = {m: {} for m in minarets}

        # ---- IoU (IDENTICAL semantics to old notebook) ----
        for tag, cam in cams.items():
            zbuf = compute_global_depth_buffer(voxel_init, cam, H, W)

            # global visible region (ALL minarets)
            pts_all = np.vstack([vox_parts[m] for m in minarets])
            pr_all = project_part_visible(
                pts_all, cam, zbuf, H, W
            )

            for m in minarets:
                gt_m = msk_parts[m].astype(bool)
                pr_m = project_part_visible(
                    vox_parts[m], cam, zbuf, H, W
                )

                gt_visible = gt_m & pr_all
                iou_vals[m][tag] = _iou_bool(gt_visible, pr_m)

        # ---- table fill ----
        for m in minarets:
            cells[m][monument] = (
                f"{iou_vals[m]['init']:.3f}→"
                f"{iou_vals[m]['rep']:.3f}→"
                f"{iou_vals[m]['final']:.3f}"
            )

        cells["Average"][monument] = (
            f"{np.mean([iou_vals[m]['init'] for m in minarets]):.3f}→"
            f"{np.mean([iou_vals[m]['rep']  for m in minarets]):.3f}→"
            f"{np.mean([iou_vals[m]['final'] for m in minarets]):.3f}"
        )

    # ---- final table ----
    df = pd.DataFrame.from_dict(cells, orient="index")
    df = df[[m for m in monuments]]
    df.columns = [MONUMENT_SHORT[m] for m in df.columns]

    print(
        """
=== Minaret IoU (INIT voxel grid)
Visualization: ALL minarets together
Table: per-minaret IoU (visible only)
Cameras: Θinit → Θkp → Θfinal
"""
    )

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=True))

    return df

def run_part_minaret_binary_iou(
    monuments,
    view,
    root_voxels,
    deformed_voxels,
    root_masks,
    cam_dir,
    part_colors,
    visualize=True,
):
    """
    Part-wise IoU + Minarets IoU + Whole (binary) IoU
    Camera: final
    init → deformed
    Visibility-aware (unchanged semantics)
    """

    PARTS = [
        "dome",
        "chhatris",
        "main_door",
        "windows",
        "plinth",
    ]

    ROWS = PARTS + ["minarets", "whole"]

    MONUMENT_SHORT = {
        "Taj": "TM",
        "Bibi": "BkM",
        "Itimad": "IuD",
        "Akbar": "AT",
        "Charminar": "CM",
    }

    cells = {r: {} for r in ROWS}

    for monument in monuments:
        print(f"\n🏛️ {monument}")

        # ---- load voxel grids ----
        voxel_init = load_voxel_grid(
            os.path.join(root_voxels, f"{monument}_voxel_grid.npz")
        )
        voxel_def = load_voxel_grid(
            os.path.join(deformed_voxels, f"{monument}_deformed_voxel_grid.npz")
        )

        # ---- load + resize mask ----
        mask_img = load_mask(
            os.path.join(
                root_masks, monument, "masks",
                f"{monument}_{view}_mask.png"
            )
        )
        mask_img = resize_mask_to_voxel_grid(mask_img, voxel_init)
        H, W = mask_img.shape[:2]

        # ---- camera (final only) ----
        cam = load_camera_json(
            os.path.join(
                cam_dir, f"{monument}_camera_params_final.json"
            ),
            view,
        )

        zbuf_init = compute_global_depth_buffer(
            voxel_init, cam, H, W
        )
        zbuf_def = compute_global_depth_buffer(
            voxel_def, cam, H, W
        )

        # ---------------- Parts ----------------
        for part in PARTS:
            mask_part = mask_parts_from_image(
                mask_img, part_colors, [part]
            )
            gt = np.any(mask_part > 0, axis=-1)

            pts_i, _ = get_voxel_points_by_parts(
                voxel_init, part_colors, [part]
            )
            pts_d, _ = get_voxel_points_by_parts(
                voxel_def, part_colors, [part]
            )

            if gt.sum() == 0 or pts_i.shape[0] == 0:
                cells[part][monument] = "--"
                continue

            pr_i = project_part_visible(
                pts_i, cam, zbuf_init, H, W
            )
            pr_d = project_part_visible(
                pts_d, cam, zbuf_def, H, W
            )

            i0 = _iou_bool(gt, pr_i)
            i1 = _iou_bool(gt, pr_d)

            if visualize:
                visualize_side_by_side(
                    gt, pr_i, pr_d, part, i0, i1
                )

            cells[part][monument] = f"{i0:.3f}→{i1:.3f}"

        # ---------------- Minarets ----------------
        pts_min, _ = get_voxel_points_by_parts(
            voxel_init, part_colors,
            ["front_minarets", "back_minarets"]
        )
        mask_min = mask_parts_from_image(
            mask_img, part_colors,
            ["front_minarets", "back_minarets"]
        )
        gt_min = np.any(mask_min > 0, axis=-1)

        pr_i = project_part_visible(
            pts_min, cam, zbuf_init, H, W
        )
        pr_d = project_part_visible(
            pts_min, cam, zbuf_def, H, W
        )

        i0 = _iou_bool(gt_min, pr_i)
        i1 = _iou_bool(gt_min, pr_d)

        if visualize:
            visualize_side_by_side(
                gt_min, pr_i, pr_d,
                "minarets", i0, i1
            )

        cells["minarets"][monument] = f"{i0:.3f}→{i1:.3f}"

        # ---------------- Whole (Binary) ----------------
        gt_whole = compute_binary_gt(mask_img, voxel_init)

        z, y, x = np.where(
            np.any(voxel_init > 0, axis=-1)
        )
        pts_i = np.stack([x, y, z], axis=1).astype(np.float32)

        z, y, x = np.where(
            np.any(voxel_def > 0, axis=-1)
        )
        pts_d = np.stack([x, y, z], axis=1).astype(np.float32)

        pr_i = project_part_visible(
            pts_i, cam, zbuf_init, H, W
        )
        pr_d = project_part_visible(
            pts_d, cam, zbuf_def, H, W
        )

        i0 = _iou_bool(gt_whole, pr_i)
        i1 = _iou_bool(gt_whole, pr_d)

        if visualize:
            visualize_side_by_side(
                gt_whole, pr_i, pr_d,
                "whole (binary)", i0, i1
            )

        cells["whole"][monument] = f"{i0:.3f}→{i1:.3f}"

    # ---------------- Final table ----------------
    df = pd.DataFrame.from_dict(cells, orient="index")
    df = df[[m for m in monuments]]
    df.columns = [MONUMENT_SHORT[m] for m in df.columns]

    print(
        """
=== Part / Minaret / Binary IoU (init → deformed)
Camera: final (Θ*)
Visibility-aware

Binary row = true whole silhouette IoU
(not average of parts)
"""
    )

    print(tabulate(df, headers="keys",
                             tablefmt="grid",
                             showindex=True))

    return df
