import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from utils.voxel_utils import *
from utils.mask_utils import *
from utils.camera_geometry import *
from utils.camera_estimation import *

from utils.visualization import *
from utils.projection_utils import *

def launch_deform_viewer_fixed_camera(
    voxel_grid, part_labels, image, cam_params, part_names,
    init_params=None  # optional dict of precomputed deform + IoU
):
    voxel_shape = voxel_grid.shape[:3]

    sliders = {
        'scale_y': widgets.FloatSlider(description='scale_y', min=0.5, max=2.0, step=0.01, value=1.0),
        'shift_y': widgets.FloatSlider(description='shift_y', min=-100, max=100, step=1.0, value=0.0),
        'scale_xz': widgets.FloatSlider(description='scale_xz', min=0.5, max=2.0, step=0.01, value=1.0),
        'shift_xz': widgets.FloatSlider(description='shift_xz', min=-100, max=100, step=1.0, value=0.0),
        'part': widgets.Dropdown(options=part_names, description='Part')
    }

    output = widgets.Output()

    # Store saved deform params
    saved_params = init_params.copy() if init_params else {}

    def deform_coords(coords, image_shape, voxel_shape, deform):
        def one_pass(c):
            center = c.mean(axis=0, keepdims=True)
            c = c - center
            H_img, W_img = image_shape
            D, H, W = voxel_shape
            pix2vox_x = W / float(W_img)
            pix2vox_y = H / float(H_img)
            pix2vox_z = D / float(W_img)
            c[:, 0] = c[:, 0] * deform['scale_xz'] + deform['shift_xz'] * pix2vox_x * np.sign(c[:, 0])
            c[:, 1] = c[:, 1] * deform['scale_y'] - deform['shift_y'] * pix2vox_y
            c[:, 2] = c[:, 2] * deform['scale_xz'] + deform['shift_xz'] * pix2vox_z * np.sign(c[:, 2])
            return np.round(c + center).astype(int)

        offsets = np.array([
            [0, 0, 0],
            [0.25, 0, 0], [-0.25, 0, 0],
            [0, 0.25, 0], [0, -0.25, 0],
            [0, 0, 0.25], [0, 0, -0.25]
        ])
        all_coords = []
        for offset in offsets:
            jittered = coords + offset
            deformed = one_pass(jittered)
            all_coords.append(deformed)

        coords_all = np.vstack(all_coords)
        coords_all = np.unique(coords_all, axis=0)
        return coords_all

    def update(_=None):
        part = sliders['part'].value
        deform = {k: sliders[k].value for k in ['scale_y', 'shift_y', 'scale_xz', 'shift_xz']}
        coords, colors = get_voxel_points_by_parts(voxel_grid, part_labels, [part])
        coords_def = deform_coords(coords.copy(), image.shape[:2], voxel_shape, deform)

        valid = (
            (coords_def[:, 0] >= 0) & (coords_def[:, 0] < voxel_shape[2]) &
            (coords_def[:, 1] >= 0) & (coords_def[:, 1] < voxel_shape[1]) &
            (coords_def[:, 2] >= 0) & (coords_def[:, 2] < voxel_shape[0])
        )
        coords_def = coords_def[valid]
        if len(coords_def) == 0:
            with output:
                clear_output(wait=True)
                print("No deformed voxels within bounds. Adjust sliders.")
            return

        colors_def = np.repeat(colors, repeats=max(1, int(len(coords_def)/len(colors))+1), axis=0)[:len(coords_def)]

        voxel_def = np.zeros_like(voxel_grid, dtype=np.uint8)
        z = coords_def[:, 2].astype(int)
        y = coords_def[:, 1].astype(int)
        x = coords_def[:, 0].astype(int)
        voxel_def[z, y, x] = colors_def.astype(np.uint8)

        proj = project_colored_voxels(
            coords_def.astype(np.float32), colors_def,
            cam_params['cam_pos'], cam_params['target'],
            cam_params['f'], cam_params['cx'], cam_params['cy'],
            H=image.shape[0], W=image.shape[1]
        )
        iou_dict, _ = compute_partwise_iou(proj, image, {part: part_labels[part]})
        iou = float(iou_dict[part])

        with output:
            clear_output(wait=True)
            visualize_voxel_projection_iou(
                voxel_def,
                {part: part_labels[part]},
                image,
                cam_params,
                mode='part_on_whole',
                save=False,
                save_root='visualisation'
            )
            print(f"{part} | IoU: {iou:.4f}")

    def run_auto_align(_):
        part = sliders['part'].value
        coords, colors = get_voxel_points_by_parts(voxel_grid, part_labels, [part])

        def iou_loss(x):
            deform = {
                'scale_y': x[0], 'shift_y': x[1],
                'scale_xz': x[2], 'shift_xz': x[3]
            }
            coords_def = deform_coords(coords.copy(), image.shape[:2], voxel_shape, deform)
            valid = (
                (coords_def[:, 0] >= 0) & (coords_def[:, 0] < voxel_shape[2]) &
                (coords_def[:, 1] >= 0) & (coords_def[:, 1] < voxel_shape[1]) &
                (coords_def[:, 2] >= 0) & (coords_def[:, 2] < voxel_shape[0])
            )
            if np.sum(valid) < 5:
                return 1e6
            coords_def = coords_def[valid]
            col = np.repeat(colors, repeats=max(1, int(len(coords_def)/len(colors))+1), axis=0)[:len(coords_def)]

            proj = project_colored_voxels(
                coords_def.astype(np.float32), col,
                cam_params['cam_pos'], cam_params['target'],
                cam_params['f'], cam_params['cx'], cam_params['cy'],
                H=image.shape[0], W=image.shape[1]
            )
            iou_dict, _ = compute_partwise_iou(proj, image, {part: part_labels[part]})
            return -iou_dict[part]

        x0 = [sliders['scale_y'].value, sliders['shift_y'].value,
              sliders['scale_xz'].value, sliders['shift_xz'].value]
        bounds = [(0.5, 2.0), (-100, 100), (0.5, 2.0), (-100, 100)]

        with output:
            clear_output(wait=True)
            print("Running auto-align using Powell...")

        res = minimize(iou_loss, x0, method='Powell', bounds=bounds, options={'maxiter': 50})

        with output:
            print("Auto-align success:", res.success)
            print("Best params:", res.x)

        sliders['scale_y'].value = res.x[0]
        sliders['shift_y'].value = res.x[1]
        sliders['scale_xz'].value = res.x[2]
        sliders['shift_xz'].value = res.x[3]

        update()

    def save_params(_):
        part = sliders['part'].value
        deform = {k: sliders[k].value for k in ['scale_y', 'shift_y', 'scale_xz', 'shift_xz']}
        coords, colors = get_voxel_points_by_parts(voxel_grid, part_labels, [part])
        coords_def = deform_coords(coords.copy(), image.shape[:2], voxel_shape, deform)

        valid = (
            (coords_def[:, 0] >= 0) & (coords_def[:, 0] < voxel_shape[2]) &
            (coords_def[:, 1] >= 0) & (coords_def[:, 1] < voxel_shape[1]) &
            (coords_def[:, 2] >= 0) & (coords_def[:, 2] < voxel_shape[0])
        )
        coords_def = coords_def[valid]
        colors_def = np.repeat(colors, repeats=max(1, int(len(coords_def)/len(colors))+1), axis=0)[:len(coords_def)]

        proj = project_colored_voxels(
            coords_def.astype(np.float32), colors_def,
            cam_params['cam_pos'], cam_params['target'],
            cam_params['f'], cam_params['cx'], cam_params['cy'],
            H=image.shape[0], W=image.shape[1]
        )
        iou_dict, _ = compute_partwise_iou(proj, image, {part: part_labels[part]})
        iou = float(iou_dict[part])
        saved_params[part] = {'deform': deform, 'iou': iou}
        with output:
            print(f"✔ Saved {part} | IoU: {iou:.4f}")

    def save_deformed_grid(_):
        """Build and save the full deformed voxel grid using all saved params."""
        voxel_def_full = np.zeros_like(voxel_grid, dtype=np.uint8)
        for part, lbl in part_labels.items():
            if part not in saved_params:
                continue
            deform = saved_params[part]['deform']
            coords, colors = get_voxel_points_by_parts(voxel_grid, part_labels, [part])
            coords_def = deform_coords(coords.copy(), image.shape[:2], voxel_shape, deform)
            valid = (
                (coords_def[:, 0] >= 0) & (coords_def[:, 0] < voxel_shape[2]) &
                (coords_def[:, 1] >= 0) & (coords_def[:, 1] < voxel_shape[1]) &
                (coords_def[:, 2] >= 0) & (coords_def[:, 2] < voxel_shape[0])
            )
            coords_def = coords_def[valid]
            if coords_def.size == 0:
                continue
            colors_def = np.repeat(colors, repeats=max(1, int(len(coords_def)/len(colors))+1), axis=0)[:len(coords_def)]
            z = coords_def[:, 2].astype(int)
            y = coords_def[:, 1].astype(int)
            x = coords_def[:, 0].astype(int)
            voxel_def_full[z, y, x] = colors_def.astype(np.uint8)

        with output:
            print("💾 Full deformed voxel grid saved.")
        return voxel_def_full

    def on_part_change(change):
        part = change['new']
        if part in saved_params:
            deform = saved_params[part]['deform']
            sliders['scale_y'].value = deform['scale_y']
            sliders['shift_y'].value = deform['shift_y']
            sliders['scale_xz'].value = deform['scale_xz']
            sliders['shift_xz'].value = deform['shift_xz']
        else:
            sliders['scale_y'].value = 1.0
            sliders['shift_y'].value = 0.0
            sliders['scale_xz'].value = 1.0
            sliders['shift_xz'].value = 0.0
        update()

    sliders['part'].observe(on_part_change, names='value')
    for k in ['scale_y', 'shift_y', 'scale_xz', 'shift_xz']:
        sliders[k].observe(update, names='value')

    auto_btn = widgets.Button(description="Auto-Align Part", button_style='info')
    save_btn = widgets.Button(description="Save Params", button_style='success')
    save_grid_btn = widgets.Button(description="Save Deformed Grid", button_style='warning')

    auto_btn.on_click(run_auto_align)
    save_btn.on_click(save_params)

    grid_storage = {'grid': None}
    def save_grid_callback(_):
        grid_storage['grid'] = save_deformed_grid(None)

    save_grid_btn.on_click(save_grid_callback)

    display(widgets.VBox([
        sliders['part'],
        widgets.HBox([sliders['scale_y'], sliders['shift_y']]),
        widgets.HBox([sliders['scale_xz'], sliders['shift_xz']]),
        widgets.HBox([auto_btn, save_btn, save_grid_btn]),
        output
    ]))

    on_part_change({'new': sliders['part'].value})
    return saved_params, grid_storage