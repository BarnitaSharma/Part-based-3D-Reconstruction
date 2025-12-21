import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from scipy.ndimage import label, binary_dilation
from scipy.optimize import minimize
from skimage.measure import regionprops, label as label2d

import ipywidgets as widgets
from IPython.display import display, clear_output

from utils.voxel_utils import *
from utils.mask_utils import *
from utils.camera_geometry import *
from utils.projection_utils import *
from utils.visualization import *


def extract_minaret_kps_for_view(
    voxel_grid,
    mask_img,
    minaret_colors,
    back_top_only=False,
):
    voxel_parts = extract_minaret_voxels_by_label(voxel_grid, minaret_colors)
    mask_parts  = extract_minaret_masks_by_label(mask_img, minaret_colors)

    common = list(set(voxel_parts) & set(mask_parts))
    if len(common) < 2:
        raise ValueError("Not enough visible minarets")

    voxel_kps = extract_top_bottom_voxel_points(
        {k: voxel_parts[k] for k in common}
    )
    image_kps = extract_top_bottom_image_points(
        {k: mask_parts[k] for k in common}
    )

    voxel_sel, image_sel = {}, {}
    for k in voxel_kps:
        m = k.split("_")[0]
        if ("1" in m) or ("2" in m and "top" in k):
            voxel_sel[k] = voxel_kps[k]
            image_sel[k] = image_kps[k]

    if len(voxel_sel) < 2:
        raise ValueError("Not enough keypoints after filtering")

    return voxel_sel, image_sel

# ============================================================
# Initial camera estimation (bbox-based)
# ============================================================

def auto_compute_initial_params_matching_bbox(voxel_grid, image, part_colors, parts_for_alignment, fov_deg=30):
    """
    Compute initial camera parameters so voxel projection roughly matches image mask bounding box.
    """
    H_img, W_img = image.shape[:2]
    voxel_pts, _ = get_voxel_points_by_parts(voxel_grid, part_colors, parts_for_alignment)
    seg_img = mask_parts_from_image(image, part_colors, parts_for_alignment)

    # Voxel 3D bounding box
    bbox_min = voxel_pts.min(axis=0)
    bbox_max = voxel_pts.max(axis=0)
    voxel_center = (bbox_min + bbox_max) / 2
    voxel_size = np.linalg.norm(bbox_max - bbox_min)

    # Image 2D mask bounding box
    mask = np.any(seg_img > 0, axis=-1)
    ys, xs = np.where(mask)
    img_bbox_min = np.array([xs.min(), ys.min()])
    img_bbox_max = np.array([xs.max(), ys.max()])
    img_bbox_center = (img_bbox_min + img_bbox_max) / 2
    img_bbox_width = np.linalg.norm(img_bbox_max - img_bbox_min)

    # Initial camera position (along +Z)
    # cam_pos = voxel_center + np.array([0, 0, voxel_size * 2.0])
    cam_pos = voxel_center + np.array([0, 0, -voxel_size * 2.0])

    target = voxel_center

    # Compute focal length from FOV and image height
    f = H_img / (2 * np.tan(np.deg2rad(fov_deg) / 2))

    # Estimate projected voxel width at that distance
    approx_voxel_proj_width = (voxel_size * f) / (voxel_size * 2.0)  # rough

    # Compute scaling factor
    scale_factor = img_bbox_width / approx_voxel_proj_width

    # Adjust focal length to match scales
    f_adjusted = f * scale_factor

    # Return adjusted params
    init_params = {
        "cam_pos": cam_pos,
        "target": target,
        "f": f_adjusted,
        "cx": W_img / 2,
        "cy": H_img / 2
    }

    print(f"Estimated scale factor: {scale_factor:.4f}")
    print(f"Adjusted focal length: {f_adjusted:.2f}")

    return init_params

def optimize_camera_with_keypoints(
    voxel_keypoints_dict, image_keypoints_dict, image, init_params, loss_type='L2'
):
    """
    Optimize full camera position and target position (look-at) and intrinsics (f, cx, cy)
    to minimize reprojection loss.
    """

    H, W = image.shape[:2]
    keys = list(image_keypoints_dict.keys())

    def loss_fn(x):
        cam_x, cam_y, cam_z, target_x, target_y, target_z, f, cx, cy = x
        cam_pos = np.array([cam_x, cam_y, cam_z])
        target = np.array([target_x, target_y, target_z])

        total = 0
        for k in keys:
            proj_pt = project(voxel_keypoints_dict[k], cam_pos, target, f, cx, cy)
            gt_pt = image_keypoints_dict[k]
            error = np.abs(proj_pt - gt_pt) if loss_type == 'L1' else (proj_pt - gt_pt) ** 2
            total += error.sum()
        return total

    # Initial parameter vector
    x0 = [
        *init_params['cam_pos'],     # cam_x, cam_y, cam_z
        *init_params['target'],      # target_x, target_y, target_z
        init_params['f'],            # f
        init_params['cx'],           # cx
        init_params['cy'],           # cy
    ]

    # Bounds (tune based on scene scale)
    bounds = [
        # (-W, 2 * W), (-H, 2 * H), (-2000, -50),     # cam_x, cam_y, cam_z
        # (-W, 2 * W), (-H, 2 * H), (-2000, 500),     # target_x, target_y, target_z
        (-W, 2 * W), (-H, 2 * H), (-2000, 100),     # cam_x, cam_y, cam_z
        (-W, 2 * W), (-H, 2 * H), (-2000,100),     # target_x, target_y, target_z
        (10, 2000),                                 # f
        (0, W), (0, H)                              # cx, cy
        # (-W, 2*W), (-H, 2*H)   
    ]

    result = minimize(loss_fn, x0, bounds=bounds, method='L-BFGS-B')
    cam_x, cam_y, cam_z, target_x, target_y, target_z, f, cx, cy = result.x

    final_params = {
        "cam_pos": np.array([cam_x, cam_y, cam_z]),
        "target": np.array([target_x, target_y, target_z]),
        "f": f,
        "cx": cx,
        "cy": cy
    }

    print("\n📷 Optimized Camera Parameters:")
    for k, v in final_params.items():
        print(f"{k}: {v}")
    print(f"📉 Final Reprojection Loss: {result.fun:.2f}")

    return final_params

# ============================================================
# Minaret extraction (unchanged, already correct)
# ============================================================

def extract_minaret_voxels_by_label(voxel_grid, minaret_colors):
    components = []

    for color in minaret_colors:
        mask = np.all(voxel_grid == color, axis=-1)
        labeled, n = label(mask)

        for cid in range(1, n + 1):
            coords = np.argwhere(labeled == cid)
            if coords.size == 0:
                continue

            centroid = coords.mean(axis=0)
            height = coords[:, 1].ptp()
            components.append((centroid, height, coords))

    if len(components) < 4:
        raise ValueError(f"Expected ≥4 minarets, found {len(components)}")

    top4 = sorted(components, key=lambda x: -x[1])[:4]
    centroids = np.stack([c[0] for c in top4])
    coord_sets = [c[2] for c in top4]

    order_x = np.argsort(centroids[:, 0])
    left, right = order_x[:2], order_x[2:]

    left = sorted(left, key=lambda i: centroids[i, 2])
    right = sorted(right, key=lambda i: centroids[i, 2])

    return {
        "LM1": coord_sets[left[0]],
        "LM2": coord_sets[left[1]],
        "RM1": coord_sets[right[0]],
        "RM2": coord_sets[right[1]],
    }


def extract_minaret_masks_by_label(image, minaret_colors):
    image_rgb = image[:, :, :3]
    masks = {}

    def regions(color):
        mask = np.all(image_rgb == color, axis=-1).astype(np.uint8)
        labeled = label2d(mask)
        return sorted(regionprops(labeled), key=lambda r: r.centroid[1])

    front = regions(minaret_colors[0])
    back  = regions(minaret_colors[1])

    if len(front) != 2:
        raise ValueError("Expected 2 front minarets")

    masks["LM1"] = (label2d(np.all(image_rgb == minaret_colors[0], axis=-1)) == front[0].label)
    masks["RM1"] = (label2d(np.all(image_rgb == minaret_colors[0], axis=-1)) == front[1].label)

    if len(back) == 2:
        masks["LM2"] = (label2d(np.all(image_rgb == minaret_colors[1], axis=-1)) == back[0].label)
        masks["RM2"] = (label2d(np.all(image_rgb == minaret_colors[1], axis=-1)) == back[1].label)
    elif len(back) == 1:
        c = back[0].centroid[1]
        f0, f1 = front[0].centroid[1], front[1].centroid[1]
        key = "LM2" if abs(c - f0) < abs(c - f1) else "RM2"
        masks[key] = (label2d(np.all(image_rgb == minaret_colors[1], axis=-1)) == back[0].label)
    else:
        raise ValueError("Unexpected number of back minarets")

    return masks


# ============================================================
# Keypoints
# ============================================================

def extract_top_bottom_voxel_points(voxel_parts):
    out = {}
    for name, vox in voxel_parts.items():
        ys = vox[:, 1]
        out[f"{name}_bottom"] = vox[ys == ys.min()].mean(axis=0)
        out[f"{name}_top"]    = vox[ys == ys.max()].mean(axis=0)
    return out


def extract_top_bottom_image_points(mask_parts):
    out = {}
    for name, mask in mask_parts.items():
        ys, xs = np.nonzero(mask)
        out[f"{name}_top"]    = (xs[ys == ys.min()].mean(), ys.min())
        out[f"{name}_bottom"] = (xs[ys == ys.max()].mean(), ys.max())
    return out

def visualize_voxel_projection_iou(
    voxel_grid,
    part_colors,
    image,
    cam_params,
    mode='part_on_whole',  # 'part_on_whole', 'part_on_part', 'whole_on_whole', 'whole_on_whole_color'
    save=False,
    save_root='visualisation'
):
    def compute_mask(image, color):
        return np.all(image == color, axis=-1)

    def compute_iou(gt_mask, prj_mask):
        inter = np.logical_and(gt_mask, prj_mask).sum()
        union = np.logical_or(gt_mask, prj_mask).sum()
        return inter / union if union > 0 else 0.0

    def outline_projection(base, gt_mask, prj_mask):
        combined_mask = np.logical_and(gt_mask, prj_mask)
        outline = binary_dilation(combined_mask) & ~combined_mask
        base[outline] = [255, 255, 0]  # yellow outline
        return base

    if save:
        save_dir = os.path.join(save_root, mode)
        os.makedirs(save_dir, exist_ok=True)

    H, W = image.shape[:2]
    combined_mask_gt = np.zeros((H, W), dtype=bool)
    combined_mask_prj = np.zeros((H, W), dtype=bool)
    combined_proj = np.zeros_like(image, dtype=np.float32)
    # BG_COLOR = np.array(part_colors["background"], dtype=np.uint8)
    BG_COLOR = np.array(part_colors.get("background", (0, 0, 0)), dtype=np.uint8)


    for part, color in part_colors.items():
        pts, col = get_voxel_points_by_parts(voxel_grid, part_colors, [part])
    
        if pts.shape[0] == 0:
            continue
    
        proj_img = project_colored_voxels(
            pts, col,
            cam_params['cam_pos'], cam_params['target'],
            cam_params['f'], cam_params['cx'], cam_params['cy'],
            H, W
        )
    
        mask_gt  = np.all(image == color, axis=-1)
        mask_prj = np.all(proj_img == color, axis=-1)
    
        combined_mask_gt  |= mask_gt
        combined_mask_prj |= mask_prj
        combined_proj     += proj_img.astype(np.float32)

   
        if mode in ['part_on_whole', 'part_on_part']:
            iou = compute_iou(mask_gt, mask_prj)

            if mode == 'part_on_whole':
                vis = (0.7 * proj_img + 0.3 * image).astype(np.uint8)

            elif mode == 'part_on_part':
                vis = np.zeros_like(image)
                vis[mask_gt] = image[mask_gt]
                vis[mask_prj] = proj_img[mask_prj]

                overlap = mask_gt & mask_prj
                blended_overlap = np.clip((0.7 * proj_f + 0.3 * image_front) * 1.5, 0, 255)

                blended_overlap = np.clip((0.7 * proj_img + 0.3 * image) * 1.5, 0, 255)
                vis[overlap] = blended_overlap[overlap].astype(np.uint8)

            vis = outline_projection(vis, mask_gt, mask_prj)

            plt.figure(figsize=(6, 6))
            plt.imshow(vis)
            plt.title(f"{part} | IoU: {iou:.3f}")
            plt.axis("off")

            if save:
                path = os.path.join(save_dir, f"{part}_overlay.png")
                plt.savefig(path)
                print(f"Saved {path}")
            plt.show()

  
    if mode == "whole_on_whole":
        combined_mask_gt = np.any(image != BG_COLOR, axis=-1)

        print("Visualizing combined binary projection vs. binary ground-truth...")
    
        overlap = combined_mask_gt & combined_mask_prj
        only_gt = combined_mask_gt & ~combined_mask_prj
        only_prj = combined_mask_prj & ~combined_mask_gt
    
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[only_gt]  = [0, 255, 0]     # GT only
        vis[only_prj] = [255, 0, 0]     # Projection only
        vis[overlap]  = [255, 255, 0]   # Overlap
    
        iou = compute_iou(combined_mask_gt, combined_mask_prj)
    
        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.title(f"Combined Binary | IoU: {iou:.3f}")
        plt.axis("off")
    
        if save:
            path = os.path.join(save_dir, "combined_binary_overlay.png")
            plt.savefig(path)
            print(f"Saved {path}")
    
        plt.show()


    if mode == 'whole_on_whole_color':
        print("Visualizing full-color projection overlay...")

        proj_img = np.clip(combined_proj, 0, 255).astype(np.uint8)
        vis = (0.7 * proj_img + 0.3 * image).astype(np.uint8)

        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.title("Combined Color Projection Overlay")
        plt.axis("off")

        if save:
            path = os.path.join(save_dir, f"combined_color_overlay.png")
            plt.savefig(path)
            print(f"Saved {path}")
        plt.show()

def launch_smart_aligner(
    voxel_grid,
    image,
    part_colors,
    parts_for_alignment=["plinth", "minarets"],
    init_params=None,
    lock_xy_equal=False
):
    H_img, W_img = image.shape[:2]
    selected_labels = {p: part_colors[p] for p in parts_for_alignment}
    seg_img = mask_parts_from_image(image, part_colors, parts_for_alignment)

    if init_params is None:
        init_params = auto_compute_initial_params_matching_bbox(
            voxel_grid, image, part_colors, parts_for_alignment
        )
    # keep original init for hard reset
    orig_init = init_params.copy()

    voxel_pts, voxel_colors = get_voxel_points_by_parts(
        voxel_grid, part_colors, parts_for_alignment
    )

    # slider factory
    def slider(name, mn, mx, val):
        return widgets.FloatSlider(description=name, min=mn, max=mx, value=val, step=1.0)

    sliders = {
        "cam_x":    slider("cam_x", -3000, 3000, init_params["cam_pos"][0]),
        "cam_y":    slider("cam_y", -3000, 3000, init_params["cam_pos"][1]),
        "cam_z":    slider("cam_z", -4000, 4000, init_params["cam_pos"][2]),
        "target_x": slider("target_x", -2000, 2000, init_params["target"][0]),
        "target_y": slider("target_y", -2000, 2000, init_params["target"][1]),
        "target_z": slider("target_z", -2000, 2000, init_params["target"][2]),
        "f":        slider("f", 100, 3000, init_params["f"]),
        # "cx":       slider("cx", 0, W_img, init_params["cx"]),
        # "cy":       slider("cy", 0, H_img, init_params["cy"]),
        "cx": slider("cx", -W_img, 2*W_img, init_params["cx"]),
        "cy": slider("cy", -H_img, 2*H_img, init_params["cy"])

    }

    random_steps   = widgets.IntSlider(description="Random Steps", min=1, max=500, value=5)
    coord_steps    = widgets.IntSlider(description="Coord Steps", min=1, max=100, value=5)
    powell_maxiter = widgets.IntSlider(description="Powell MaxIter", min=1, max=100, value=5)

    output = widgets.Output()
    saved_params = {}

    def get_params():
        cam = np.array([sliders[f"cam_{c}"].value for c in "xyz"])
        tgt = np.array([sliders[f"target_{c}"].value for c in "xyz"])
        if lock_xy_equal:
            cam[0], cam[1] = tgt[0], tgt[1]
        return {
            "cam_pos": cam,
            "target":  tgt,
            "f":        sliders["f"].value,
            "cx":       sliders["cx"].value,
            "cy":       sliders["cy"].value,
            "H":        H_img,
            "W":        W_img
        }

    def set_params(p):
        for c, v in zip("xyz", p["cam_pos"]):
            sliders[f"cam_{c}"].value = v
        for c, v in zip("xyz", p["target"]):
            sliders[f"target_{c}"].value = v
        sliders["f"].value  = p["f"]
        sliders["cx"].value = p["cx"]
        sliders["cy"].value = p["cy"]

    def quick_overlay_proj(cam_params, title=""):
        proj = project_colored_voxels(
            voxel_pts, voxel_colors,
            cam_params["cam_pos"], cam_params["target"],
            cam_params["f"], cam_params["cx"], cam_params["cy"],
            cam_params["H"], cam_params["W"]
        )
        # proj = np.flipud(proj)

        _, iou = compute_partwise_iou(proj, seg_img, selected_labels)
        overlay = seg_img.copy()
        mask = np.any(proj > 0, axis=-1)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * proj[mask]).astype(np.uint8)
        with output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plt.imshow(overlay)
            plt.title(f"{title} | IoU: {iou:.4f}")
            plt.axis("off")
            plt.show()
        return iou

    def to_vector(p):
        if lock_xy_equal:
            return np.array([p["cam_pos"][2], p["target"][2], p["f"], p["cx"], p["cy"]])
        else:
            return np.concatenate([p["cam_pos"], p["target"], [p["f"], p["cx"], p["cy"]]])

    def from_vector(x):
        if lock_xy_equal:
            tx, ty = sliders["target_x"].value, sliders["target_y"].value
            return {
                "cam_pos": np.array([tx, ty, x[0]]),
                "target":  np.array([tx, ty, x[1]]),
                "f":       x[2], "cx": x[3], "cy": x[4],
                "H":       H_img, "W": W_img
            }
        else:
            return {
                "cam_pos": x[:3],
                "target":  x[3:6],
                "f":       x[6], "cx": x[7], "cy": x[8],
                "H":       H_img, "W": W_img
            }

    def evaluate(p):
        proj = project_colored_voxels(
            voxel_pts, voxel_colors,
            p["cam_pos"], p["target"], p["f"], p["cx"], p["cy"], p["H"], p["W"]
        )
        _, iou = compute_partwise_iou(proj, seg_img, selected_labels)
        return -iou

    # ----- OPTIMIZERS -----
    def run_random(_):
        base = get_params()
        best_iou = quick_overlay_proj(base, "Random Init")
        best_p = base.copy()
    
        step_sizes = {
            "cam_pos":  np.array([50, 50, 100]),
            "target":   np.array([50, 50, 100]),
            "f":        50,
            "cx":       20,
            "cy":       20
        }
    
        for i in range(random_steps.value):
            trial = base.copy()
            trial["cam_pos"] = base["cam_pos"] + np.random.uniform(-1, 1, 3) * step_sizes["cam_pos"]
            trial["target"]  = base["target"]  + np.random.uniform(-1, 1, 3) * step_sizes["target"]
            trial["f"]       = base["f"]       + np.random.uniform(-1, 1) * step_sizes["f"]
            trial["cx"]      = base["cx"]      + np.random.uniform(-1, 1) * step_sizes["cx"]
            trial["cy"]      = base["cy"]      + np.random.uniform(-1, 1) * step_sizes["cy"]
            if lock_xy_equal:
                trial["cam_pos"][:2] = trial["target"][:2]
    
            # Compute param changes
            changes = []
            for idx, name in enumerate("xyz"):
                d = trial["cam_pos"][idx] - base["cam_pos"][idx]
                if abs(d) > 1e-1:
                    changes.append(f"cam_{name}{d:+.1f}")
            for idx, name in enumerate("xyz"):
                d = trial["target"][idx] - base["target"][idx]
                if abs(d) > 1e-1:
                    changes.append(f"target_{name}{d:+.1f}")
            for key in ["f", "cx", "cy"]:
                d = trial[key] - base[key]
                if abs(d) > 1e-1:
                    changes.append(f"{key}{d:+.1f}")
    
            label = " ".join(changes) or "no_change"
            iou = quick_overlay_proj(trial, f"Random {i+1} | {label}")
            if iou > best_iou:
                best_iou, best_p = iou, trial.copy()
    
        set_params(best_p)
        print(f"Random Done | Best IoU: {best_iou:.4f}")

    def run_coord(_):
        base = get_params()
        best_iou = quick_overlay_proj(base, "Coord Init")
        best_p   = base.copy()
        keys = list(sliders.keys())
    
        for rnd in range(coord_steps.value):
            improved = False
            for k in keys:
                for delta in (-20, 20):
                    trial = best_p.copy()
                    if k.startswith("cam_") and not lock_xy_equal:
                        trial["cam_pos"]["xyz".index(k[-1])] += delta
                    elif k.startswith("target_"):
                        trial["target"]["xyz".index(k[-1])] += delta
                        if lock_xy_equal and k in ("target_x","target_y"):
                            trial["cam_pos"]["xyz".index(k[-1])] += delta
                    elif k in ("f", "cx", "cy"):
                        trial[k] += delta
                    else:
                        continue
    
                    param = k
                    title = f"CD {rnd+1} | {param}{delta:+.1f}"
                    iou = quick_overlay_proj(trial, title)
    
                    if iou > best_iou:
                        best_iou, best_p = iou, trial.copy()
                        improved = True
                        break
                if improved:
                    break
    
        set_params(best_p)
        print(f"Coord Descent Done | Best IoU: {best_iou:.4f}")

    def run_powell(_):
        base = get_params()
        x0 = to_vector(base)
        names = (
            ["cam_z", "target_z", "f", "cx", "cy"]
            if lock_xy_equal else
            ["cam_x", "cam_y", "cam_z", "target_x", "target_y", "target_z", "f", "cx", "cy"]
        )
    
        calls = {"n": 0}
        def obj(x):
            calls["n"] += 1
            p = from_vector(x)
            if calls["n"] % 5 == 0:
                deltas = x - obj.prev_x
                changes = [f"{names[i]}{d:+.1f}" for i, d in enumerate(deltas) if abs(d) > 1e-1]
                obj.prev_x = x.copy()
                label = " ".join(changes) or "no_change"
                quick_overlay_proj(p, f"Powell Eval {calls['n']} | {label}")
            return evaluate(p)
    
        obj.prev_x = x0.copy()
    
        res = minimize(
            obj, x0, method='Powell',
            options={
                'maxiter': powell_maxiter.value,
                'maxfev':  powell_maxiter.value * 10,
                'xtol':    1e-3,
                'ftol':    1e-3,
                'disp':    True
            }
        )
    
        p = from_vector(res.x)
        iou = quick_overlay_proj(p, "Powell Final")
        set_params(p)
        print(f"Powell Done | Best IoU: {iou:.4f}")

    # save/load
    def on_save(_):
        saved_params.clear()
        saved_params.update(get_params())
        print("✔ Saved")
    def on_load(_):
        set_params(saved_params)
        quick_overlay_proj(get_params(), "Loaded Params")

    # init reset
    def on_init(_):
        set_params(orig_init)
        quick_overlay_proj(get_params(), "Reset to Init")

    for s in sliders.values():
        s.observe(lambda _: quick_overlay_proj(get_params(), "Live"), names="value")

    buttons = [
        ("Random Search","warning",run_random),
        ("Coordinate Descent","info",run_coord),
        ("Powell","success",run_powell),
        ("Save","",on_save),
        ("Load","",on_load),
    ]
    button_widgets = [widgets.Button(description=l, button_style=b) for l,b,_ in buttons]
    for btn,(_,_,cb) in zip(button_widgets, buttons): btn.on_click(cb)
    init_btn = widgets.Button(description="Init", button_style="")
    init_btn.on_click(on_init)

    layout = widgets.VBox([
        widgets.HBox([sliders["f"]]),
        widgets.HBox([sliders[k] for k in ["cam_x","cam_y","cam_z"]]),
        widgets.HBox([sliders[k] for k in ["target_x","target_y","target_z"]]),
        widgets.HBox([sliders["cx"], sliders["cy"]]),
        widgets.HBox([init_btn] + button_widgets),
        widgets.HBox([random_steps, coord_steps, powell_maxiter]),
        output
    ])

    display(layout)
    quick_overlay_proj(get_params(), "Initial Projection")
    return saved_params

def compute_partwise_iou(proj_mask, gt_mask, part_colors):
    flat_proj = proj_mask.reshape(-1, 3)
    flat_gt   = gt_mask.reshape(-1, 3)

    iou_per_part = {}
    total = []

    for part, color in part_colors.items():
        proj_part = np.all(flat_proj == color, axis=1)
        gt_part   = np.all(flat_gt == color, axis=1)

        inter = np.logical_and(proj_part, gt_part).sum()
        union = np.logical_or(proj_part, gt_part).sum()
        iou = inter / union if union > 0 else 0.0

        iou_per_part[part] = iou
        total.append(iou)

    return iou_per_part, np.mean(total)