from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np
import ipywidgets as W
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path as MplPath
import cv2


def launch_ui(*, ROOT, EXTS, TARGET_CANVAS_W, LABEL_PARTS, sam_predictor):
    sam_predictor = sam_predictor

    
    # ===== helpers =====
    def list_monuments():
        if not ROOT.exists():
            return []
        dirs = [p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")]
        return [(p.name, p) for p in sorted(dirs, key=lambda x: x.name)]

    def list_images(folder_path: Optional[Path]):
        """
        Expects structure:
        data/
          Monument/
            images/
              *.jpg
        """
        if not folder_path:
            return []
    
        images_dir = folder_path / "images"
        if not images_dir.exists():
            return []
    
        files = [
            p for p in images_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in EXTS
        ]
        return [(p.name, p) for p in sorted(files, key=lambda x: x.name)]

    def pil_to_np(p): return np.array(p.convert("RGB"))
    def np_to_pil(a): return Image.fromarray(a.astype(np.uint8), mode="RGB")
    def clamp(v, lo, hi): return max(lo, min(hi, v))

    def find_symmetry_axis(img_np):
        H, W = img_np.shape[:2]
        gray = img_np.mean(axis=2).astype(np.float32)
        start, end = W // 4, 3 * W // 4
        best_col, min_err = W // 2, float('inf')
        for col in range(start, end):
            left  = gray[:, :col][:, ::-1]
            right = gray[:, col:]
            mw = min(left.shape[1], right.shape[1])
            if mw <= 0: continue
            err = np.abs(left[:, :mw] - right[:, :mw]).mean()
            if err < min_err: min_err, best_col = err, col
        return int(best_col)

    # ===== widgets =====
    mon_dd   = W.Dropdown(options=list_monuments(), description="Monument:", layout=W.Layout(width="300px"))
    img_dd   = W.Dropdown(options=[], description="Image:", layout=W.Layout(width="400px"))
    load_btn = W.Button(description="Load", button_style="primary")
    refresh_btn = W.Button(description="Refresh", button_style="info")

    sym_toggle = W.ToggleButtons(options=[("Sym ON","on"),("Sym OFF","off")], value="off", description="Sym:")
    sym_auto   = W.Button(description="Auto axis")
    sym_x_box  = W.BoundedIntText(value=0, min=0, max=999999, step=1, description="Axis x:")

    crop_btn = W.Button(description="Crop", button_style="success")
    undo_btn = W.Button(description="Undo")
    reset_btn= W.Button(description="Reset")
    save_btn = W.Button(description="Save crop")

    sam_label_dd = W.Dropdown(options=list(LABEL_PARTS.keys()), value="full_building",
                              description="SAM label:", layout=W.Layout(width="220px"))
    sam_mode = W.ToggleButtons(options=[("Off","off"),("Click","click"),("Box","bbox")],
                               value="off", description="SAM:")
    sam_alpha = W.FloatSlider(value=0.5, min=0.05, max=0.9, step=0.05,
                              description="Alpha:", readout=True, layout=W.Layout(width="280px"))
    sam_undo_btn = W.Button(description="SAM Undo")
    sam_reset_btn= W.Button(description="SAM Reset")
    sam_save_btn = W.Button(description="Save overlay")

    sam_apply_mode = W.ToggleButtons(
        options=[("Add (no overwrite)", "add"), ("Replace (overwrite)", "replace"), ("Subtract (erase)", "subtract")],
        value="add", description="Apply:"
    )
    mc_close_btn = W.Button(description="Close Holes")
    mc_remove_btn = W.Button(description="Remove Small")
    mc_kernel = W.IntSlider(value=5, min=3, max=21, step=2, description="Kernel")
    mc_min_area = W.IntSlider(value=50, min=1, max=5000, step=1, description="Min area")

    mask_lasso_toggle = W.ToggleButton(value=False, description="Mask Lasso",
                                       tooltip="Draw polygon on right mask to edit")

    status = W.HTML("<code>Ready.</code>", layout=W.Layout(margin="6px 0 0 0"))

    # ===== canvas =====
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.resample'] = False
    plt.rcParams['figure.constrained_layout.use'] = False

    fig, ax = plt.subplots()
    ax.set_axis_off()

    ax_mask = fig.add_axes([0.52, 0.02, 0.46, 0.96])
    ax_mask.set_axis_off()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
    except Exception:
        pass

    canvas = fig.canvas
    canvas.layout.display = "none"
    canvas.layout.width = "1px"
    canvas.layout.height = "1px"
    canvas.layout.min_width = "1px"
    canvas.layout.min_height = "1px"

    ROW_LAYOUT = W.Layout(
    width="100%",
    display="flex",
    flex_flow="row nowrap",
    align_items="center",
    justify_content="flex-start",
    gap="8px",
    )

    row1 = W.HBox([refresh_btn, mon_dd, img_dd, load_btn], layout=ROW_LAYOUT)
    row2 = W.HBox([sym_toggle, sym_auto, sym_x_box, crop_btn, undo_btn, reset_btn, save_btn], layout=ROW_LAYOUT)
    row3 = W.HBox([sam_label_dd, sam_mode, sam_alpha, sam_undo_btn, sam_reset_btn, sam_save_btn], layout=ROW_LAYOUT)
    row4 = W.HBox([sam_apply_mode, mc_close_btn, mc_remove_btn, mc_kernel, mc_min_area, mask_lasso_toggle], layout=ROW_LAYOUT)

    ui = W.VBox([row1, row2, row3, row4, canvas, status])
    display(ui)
    canvas.layout.display = ""
    def set_status(msg): status.value = f"<code>{msg}</code>"
    
    # ===== state =====
    img_artist = None           # base image (left)
    overlay_artist = None       # SAM overlay RGBA (left)
    mask_artist = None          # color mask image (right)
    sym_line = None
    rs_crop = None              # crop selector (left)
    rs_sam = None               # SAM bbox selector (left)
    mask_lasso = None           # LassoSelector attached to ax_mask
    last_crop_np = None
    
    class S:
        path: Optional[Path] = None
        base_np = None
        curr_np = None
        sym_on = False
        sym_x = None
        undo = []
    
    # ===== SAM state (with last-action-wins draw order) =====
    seg_masks = None            # dict[str] -> np.uint8 mask (H,W)
    sam_undo_stack = []         # stack of (masks_copy, draw_order_copy)
    draw_order = []             # label names; later entries render on top
    
    _last_canvas_shape = [None]
    _last_image_shape  = [None]
    
    # ===== exported artifacts =====
    last_color_mask_np = None   # HxWx3 RGB, updated with current multi-colored mask
    
    # ===== sizing / draw =====
    def _resize_canvas_for(img_np):
        H, W = img_np.shape[:2]
        target_w = max(400, min(1400, TARGET_CANVAS_W))
        target_h = int(target_w * (H / W))
        canvas.layout.width  = f"{target_w}px"
        canvas.layout.height = f"{target_h}px"
    
    def _set_canvas_size_if_needed(img_np):
        H, W = img_np.shape[:2]
        if _last_canvas_shape[0] != (H, W):
            _last_canvas_shape[0] = (H, W)
            _resize_canvas_for(img_np)
    
    def _ensure_overlay_artist():
        nonlocal overlay_artist
        if overlay_artist is None:
            H, W = S.curr_np.shape[:2]
            empty = np.zeros((H, W, 4), np.uint8)
            overlay_artist = ax.imshow(empty, interpolation='nearest', resample=False)
            overlay_artist.set_zorder(10)
    
    def draw_img(img_np):
        """Left = image+overlay, Right = mask preview."""
        nonlocal img_artist, mask_artist
        if img_np is None:
            return
        if canvas.layout.display == "none":
            canvas.layout.display = ""  # show in place
        _set_canvas_size_if_needed(img_np)
    
        # Lay out left (image) and right (mask) halves
        ax.set_axis_off()
        ax.set_position([0.02, 0.02, 0.46, 0.96])      # LEFT pane
        ax_mask.set_axis_off()
        ax_mask.set_position([0.52, 0.02, 0.46, 0.96]) # RIGHT pane
    
        H, W = img_np.shape[:2]
        first = img_artist is None
        shape_changed = _last_image_shape[0] != (H, W)
        _last_image_shape[0] = (H, W)
    
        if first:
            img_artist = ax.imshow(img_np, interpolation='nearest', resample=False)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            fig.canvas.draw()
        else:
            img_artist.set_data(img_np)
            if shape_changed:
                img_artist.set_extent((0, W, H, 0))
                ax.set_xlim(0, W); ax.set_ylim(H, 0)
            fig.canvas.draw_idle()
    
        _ensure_overlay_artist()
        if shape_changed and overlay_artist is not None:
            overlay_artist.set_extent((0, W, H, 0))
    
        # Reset right panel image when a new base image or size arrives
        ax_mask.cla(); ax_mask.set_axis_off()
        mask_artist = None
        fig.canvas.draw_idle()
    
    def set_sym_line_visible(visible: bool):
        nonlocal sym_line
        if visible and S.sym_x is not None and S.curr_np is not None:
            if sym_line is None or sym_line.axes is None:
                sym_line = ax.axvline(x=S.sym_x, linestyle='--', color='red')
            sym_line.set_xdata([S.sym_x, S.sym_x])
            sym_line.set_visible(True)
        else:
            if sym_line is not None:
                sym_line.set_visible(False)
        fig.canvas.draw_idle()
    
    # ===== selectors =====
    def _destroy_selectors():
        nonlocal rs_crop, rs_sam, mask_lasso
        if rs_crop is not None:
            try: rs_crop.set_active(False)
            except: pass
            try: rs_crop.disconnect_events()
            except: pass
            rs_crop = None
        if rs_sam is not None:
            try: rs_sam.set_active(False)
            except: pass
            try: rs_sam.disconnect_events()
            except: pass
            rs_sam = None
        if mask_lasso is not None:
            try: mask_lasso.disconnect_events()
            except: pass
            mask_lasso = None
            mask_lasso_toggle.value = False
    
    def _ensure_crop_selector():
        nonlocal rs_crop
        if rs_crop is not None:
            rs_crop.set_active(sam_mode.value == "off")
            return
        rs_crop = RectangleSelector(ax, onselect=lambda *_: None,
                                    useblit=False, interactive=True,
                                    minspanx=5, minspany=5, spancoords='pixels')
        rs_crop.set_active(sam_mode.value == "off")
    
    def _ensure_sam_selector():
        nonlocal rs_sam
        if rs_sam is not None:
            rs_sam.set_active(sam_mode.value == "bbox")
            return
        def _on_select(eclick, erelease):
            if sam_mode.value != "bbox": return
            x0, y0 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
            x1, y1 = int(max(eclick.xdata, erelease.xdata)), int(max(erelease.ydata, eclick.ydata))
            _apply_sam_bbox((x0, y0, x1, y1))
        rs_sam = RectangleSelector(ax, onselect=_on_select,
                                   useblit=False, interactive=False,
                                   minspanx=5, minspany=5, spancoords='pixels')
        rs_sam.set_active(sam_mode.value == "bbox")
    
    def _crop_extents():
        if rs_crop is None: return None
        x0, x1 = rs_crop.extents[0], rs_crop.extents[1]
        y0, y1 = rs_crop.extents[2], rs_crop.extents[3]
        if None in (x0, x1, y0, y1): return None
        L = int(min(x0, x1)); R = int(max(x0, x1))
        T = int(min(y0, y1)); B = int(max(y0, y1))
        return L, T, R, B
    
    # ===== SAM overlay + cleaning helpers =====
    def _init_seg_masks(shape_hw):
        """Initialize masks and reset draw order to default label order."""
        nonlocal seg_masks, sam_undo_stack, draw_order
        H, W = shape_hw
        seg_masks = {k: np.zeros((H, W), np.uint8) for k in LABEL_PARTS}
        draw_order = list(LABEL_PARTS.keys())
        sam_undo_stack = []
    
    def _push_sam_undo():
        """Save a snapshot of masks and draw order for undo."""
        sam_undo_stack.append((
            {k: v.copy() for k, v in seg_masks.items()},
            draw_order.copy()
        ))
    
    def _render_overlay_rgba(alpha: float):
        """Build an RGBA overlay from current masks, following draw_order (last action wins)."""
        if S.curr_np is None: return None
        H, W = S.curr_np.shape[:2]
        ov = np.zeros((H, W, 4), np.uint8)
        a = int(round(alpha * 255))
        for name in draw_order:  # last entries draw on top
            m = seg_masks[name]
            if not np.any(m): continue
            color = np.array(LABEL_PARTS[name], dtype=np.uint8)
            idx = m.astype(bool)
            ov[idx, :3] = color
            ov[idx,  3] = a
        return ov
    
    def _render_color_mask_rgb():
        """Return HxWx3 uint8 RGB composite using draw_order (last action wins)."""
        if S.curr_np is None:
            return None
        H, W = S.curr_np.shape[:2]
        rgb = np.zeros((H, W, 3), np.uint8)
        for name in draw_order:
            m = seg_masks[name]
            if not np.any(m):
                continue
            rgb[m.astype(bool)] = np.array(LABEL_PARTS[name], dtype=np.uint8)
        return rgb
    
    def _update_overlay():
        """Update left overlay and right mask preview."""
        nonlocal last_color_mask_np, mask_artist
        if overlay_artist is None or S.curr_np is None: return
        ov = _render_overlay_rgba(sam_alpha.value)
        if ov is None: return
        overlay_artist.set_data(ov)
    
        # Right panel: live color mask
        last_color_mask_np = _render_color_mask_rgb()
        if last_color_mask_np is not None:
            if mask_artist is None or mask_artist.get_array().shape[:2] != last_color_mask_np.shape[:2]:
                ax_mask.cla(); ax_mask.set_axis_off()
                mask_artist = ax_mask.imshow(last_color_mask_np, interpolation='nearest', resample=False)
                ax_mask.set_title("Current mask")
            else:
                mask_artist.set_data(last_color_mask_np)
        fig.canvas.draw_idle()
    
    # --- cleaning helpers (binary on current label) ---
    def _binary_for_label(label_name):
        return seg_masks[label_name].astype(np.uint8)
    
    def _ensure_odd(n):  # ensure odd kernel size
        n = int(n)
        return n if n % 2 == 1 else n + 1
    
    def _close_holes_binary(mask_bin, ksize):
        k = _ensure_odd(max(3, int(ksize)))
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    
    def _remove_small_regions_binary(mask_bin, min_area):
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        keep = np.zeros_like(mask_bin)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
                keep[labels == i] = 1
        return keep
    
    # ===== APPLY new mask to a label with modes: add/replace/subtract =====
    def _apply_mask_to_label(full_mask_bool, label_name):
        """
        full_mask_bool: HxW bool in current image resolution from SAM or lasso
        Modes:
          - add:      add only to pixels not owned by any label
          - replace:  claim pixels for label_name (clear them from other labels)
          - subtract: remove pixels from label_name
        """
        nonlocal draw_order
        fm = full_mask_bool.astype(bool)
        if not np.any(fm):
            return
    
        _push_sam_undo()
    
        mode = sam_apply_mode.value
        if mode == "subtract":
            seg_masks[label_name][fm] = 0
        elif mode == "add":
            occupied = np.zeros_like(seg_masks[label_name], dtype=bool)
            for _, m in seg_masks.items():
                occupied |= m.astype(bool)
            allowed = fm & (~occupied)
            seg_masks[label_name] = (seg_masks[label_name].astype(bool) | allowed).astype(np.uint8)
        elif mode == "replace":
            for k in seg_masks.keys():
                if k != label_name:
                    seg_masks[k][fm] = 0
            seg_masks[label_name] = (seg_masks[label_name].astype(bool) | fm).astype(np.uint8)
    
        # last-action-wins
        if label_name in draw_order:
            try: draw_order.remove(label_name)
            except ValueError: pass
        draw_order.append(label_name)
    
        _update_overlay()
    
    # ===== shared filename helper =====
    def _current_bbox_for_filename():
        """Return (L,T,R,B) using current crop selector extents if present; otherwise full image box."""
        if S.curr_np is None:
            return 0, 0, 0, 0
        ext = _crop_extents()
        if ext:
            L, T, R, B = ext
        else:
            H, W = S.curr_np.shape[:2]
            L, T, R, B = 0, 0, W, H
        return int(L), int(T), int(R), int(B)
    
    # ===== actions (file + crop) =====
    def refresh_lists(_=None):
        mons = list_monuments()
        mon_dd.options = mons
        mon_dd.value = mons[0][1] if mons else None
        if mon_dd.value:
            imgs = list_images(mon_dd.value)
            img_dd.options = imgs
            img_dd.value = imgs[0][1] if imgs else None
            set_status(f"{mon_dd.value.name}: {len(imgs)} file(s)")
        else:
            img_dd.options = []; img_dd.value = None
            set_status(f"ROOT not found or empty: {ROOT}")
    
    def on_mon_change(change):
        folder = change["new"]
        imgs = list_images(folder)
        img_dd.options = []; img_dd.value = None
        img_dd.options = imgs
        img_dd.value = imgs[0][1] if imgs else None
        set_status(f"{folder.name}: {len(imgs)} file(s)")
    
    def do_load(_=None):
        nonlocal img_artist, last_crop_np, mask_artist
        img_path: Optional[Path] = img_dd.value
        if img_path is None: return set_status("Pick a monument + image.")
        if not img_path.exists(): return set_status(f"File not found: {img_path}")
    
        S.path = img_path
        im = Image.open(S.path).convert("RGB")
        S.base_np = pil_to_np(im)
        S.curr_np = S.base_np.copy()
    
        # reset visuals
        canvas.layout.display = ""
        img_artist = None
        mask_artist = None
    
        # remove old selectors bound to old axes
        _destroy_selectors()
    
        ax.cla(); ax.set_axis_off()
        ax_mask.cla(); ax_mask.set_axis_off()
        draw_img(S.curr_np)
    
    
    
        # symmetry
        S.sym_x = find_symmetry_axis(S.curr_np)
        sym_x_box.max = S.curr_np.shape[1]
        sym_x_box.value = int(S.sym_x)
        set_sym_line_visible(S.sym_on)
    
        # selectors
        _ensure_crop_selector()
        _ensure_sam_selector()
        rs_crop.set_active(True)
        rs_sam.set_active(False)
        sam_mode.value = "off"   # default back to crop mode
        mask_lasso_toggle.value = False  # start off
    
        # reset undo + SAM masks
        S.undo.clear()
        _init_seg_masks(S.curr_np.shape[:2])
    
        last_crop_np = None
        H, W = S.curr_np.shape[:2]
        _update_overlay()
        set_status(f"Loaded {S.path.name} ({W}x{H}), sym_x={S.sym_x}")
    
    def do_crop(_=None):
        nonlocal last_crop_np
        if S.curr_np is None: return set_status("Load an image first.")
        ext = _crop_extents()
        if not ext: return set_status("Left-drag a rectangle to crop.")
        H, W = S.curr_np.shape[:2]
        L, T, R, B = ext
        L, T = clamp(L, 0, W), clamp(T, 0, H)
        R, B = clamp(R, 0, W), clamp(B, 0, H)
        if R-L <= 0 or B-T <= 0: return set_status("Zero-area selection.")
    
        if S.sym_on and S.sym_x is not None:
            half = max(S.sym_x - L, R - S.sym_x)
            L = clamp(S.sym_x - half, 0, W)
            R = clamp(S.sym_x + half, 0, W)
            sym_info = f" (sym around x={S.sym_x})"
        else:
            sym_info = ""
    
        S.undo.append(S.curr_np.copy())
        crop = S.curr_np[T:B, L:R]
        S.curr_np = crop
        last_crop_np = crop
    
        if S.sym_on and S.sym_x is not None:
            S.sym_x = clamp(S.sym_x - L, 0, S.curr_np.shape[1])
    
        draw_img(S.curr_np)
    
        # --- symmetry line should not persist after crop ---
        S.sym_on = False
        set_sym_line_visible(False)
    
        # --- remove crop box + sam box + lasso ---
        _destroy_selectors()
        _ensure_crop_selector()
    
        _init_seg_masks(S.curr_np.shape[:2])
        _update_overlay()
        set_status(f"Cropped → {crop.shape}{sym_info}")
        
    def do_undo(_=None):
        if not S.undo: return set_status("Nothing to undo.")
        prev = S.undo.pop()
        S.curr_np = prev
        draw_img(prev)
        set_sym_line_visible(S.sym_on)
        _init_seg_masks(S.curr_np.shape[:2])
        _update_overlay()
        set_status(f"Undo → {prev.shape}")
    
    def do_reset(_=None):
        if S.base_np is None: return set_status("Nothing to reset.")
        S.curr_np = S.base_np.copy()
        S.sym_x = find_symmetry_axis(S.curr_np)
        draw_img(S.curr_np)
        set_sym_line_visible(S.sym_on)
        S.undo.clear()
        _init_seg_masks(S.curr_np.shape[:2])
        _update_overlay()
        set_status("Reset.")
    
    def do_save(_=None):
        if S.curr_np is None or S.path is None:
            return set_status("Nothing to save.")
        crops_dir = S.path.parent / "crops"
        crops_dir.mkdir(exist_ok=True)
        L, T, R, B = _current_bbox_for_filename()
        out_name = f"{S.path.stem}_crop_{L}_{T}_{R}_{B}{S.path.suffix}"
        out_path = crops_dir / out_name
        np_to_pil(S.curr_np).save(out_path)
        set_status(f"Saved → {out_path}")
    
    # ===== symmetry + wiring =====
    def on_sym_toggle(change):
        S.sym_on = (change["new"] == "on")
        set_sym_line_visible(S.sym_on)
        set_status("Symmetry ON" if S.sym_on else "Symmetry OFF")
    
    def on_sym_auto(_=None):
        if S.curr_np is None: return set_status("Load first.")
        S.sym_x = find_symmetry_axis(S.curr_np)
        sym_x_box.max = S.curr_np.shape[1]
        sym_x_box.value = int(S.sym_x)
        set_sym_line_visible(S.sym_on)
        set_status(f"Auto symmetry x={S.sym_x}")
    
    def on_sym_x(change):
        if S.curr_np is None: return
        S.sym_x = clamp(int(change["new"]), 0, S.curr_np.shape[1])
        set_sym_line_visible(S.sym_on)
        set_status(f"Set symmetry x={S.sym_x}")
    
    refresh_btn.on_click(refresh_lists)
    mon_dd.observe(on_mon_change, names="value")
    load_btn.on_click(do_load)
    crop_btn.on_click(do_crop)
    undo_btn.on_click(do_undo)
    reset_btn.on_click(do_reset)
    save_btn.on_click(do_save)
    sym_toggle.observe(on_sym_toggle, names="value")
    sym_auto.on_click(on_sym_auto)
    sym_x_box.observe(on_sym_x, names="value")
    
    # ===== SAM: interactions (unchanged) =====
    def _apply_sam_point(x, y, pos_label=1):
        if S.curr_np is None:
            return set_status("Load an image first.")
        # if 'sam_predictor' not in nonlocals():
        #     return set_status("sam_predictor not found. Please create it before using SAM.")
        sam_predictor.set_image(S.curr_np.copy())
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[int(x), int(y)]], dtype=np.float32),
            point_labels=np.array([int(pos_label)], dtype=np.int32),
            multimask_output=True
        )
        best = masks[np.argmax(scores)]
        _apply_mask_to_label(best.astype(bool), sam_label_dd.value)
        set_status(f"SAM point ({int(x)},{int(y)}), {'pos' if pos_label==1 else 'neg'} → {sam_label_dd.value}")
    
    def _apply_sam_bbox(rect_xyxy):
        if S.curr_np is None:
            return set_status("Load an image first.")
        # if 'sam_predictor' not in nonlocals():
        #     return set_status("sam_predictor not found. Please create it before using SAM.")
        x0, y0, x1, y1 = rect_xyxy
        if x1-x0 <= 2 or y1-y0 <= 2:
            return set_status("SAM box too small.")
        box = np.array([int(x0), int(y0), int(x1), int(y1)], dtype=np.int32)
        sam_predictor.set_image(S.curr_np.copy())
        masks, scores, _ = sam_predictor.predict(box=box[None, :], multimask_output=True)
        best = masks[np.argmax(scores)]
        _apply_mask_to_label(best.astype(bool), sam_label_dd.value)
        set_status(f"SAM box {tuple(box.tolist())} → {sam_label_dd.value}")
    
    def _on_canvas_click(event):
        if sam_mode.value != "click": return
        if event.inaxes != ax: return
        if event.xdata is None or event.ydata is None: return
        is_shift = (event.key == 'shift') or (event.key == 'shift+control')
        pos = 0 if is_shift else 1
        _apply_sam_point(event.xdata, event.ydata, pos_label=pos)
    
    cid_click = fig.canvas.mpl_connect('button_press_event', _on_canvas_click)
    
    def on_sam_mode(change):
        mode = change["new"]
        if rs_crop is not None:
            rs_crop.set_active(mode == "off")
        _ensure_sam_selector()
        rs_sam.set_active(mode == "bbox")
        set_status(f"SAM mode → {mode}")
    sam_mode.observe(on_sam_mode, names='value')
    
    def on_sam_alpha(change):
        _update_overlay()
    sam_alpha.observe(on_sam_alpha, names='value')
    
    def on_sam_undo(_):
        nonlocal seg_masks, draw_order
        if not sam_undo_stack:
            return set_status("SAM: nothing to undo.")
        prev_masks, prev_order = sam_undo_stack.pop()
        seg_masks = prev_masks
        draw_order = prev_order
        _update_overlay()
        set_status("SAM: undid last action.")
    sam_undo_btn.on_click(on_sam_undo)
    
    def on_sam_reset(_):
        _init_seg_masks(S.curr_np.shape[:2])
        _update_overlay()
        set_status("SAM: reset masks.")
    sam_reset_btn.on_click(on_sam_reset)
    
    # ===== Mask cleaning & lasso wiring =====
    def on_mc_close(_):
        if S.curr_np is None: return set_status("Load an image first.")
        lbl = sam_label_dd.value
        _push_sam_undo()
        m = _binary_for_label(lbl)
        seg_masks[lbl] = _close_holes_binary(m, mc_kernel.value).astype(np.uint8)
        _update_overlay()
        set_status(f"Closed holes → {lbl} (k={mc_kernel.value})")
    mc_close_btn.on_click(on_mc_close)
    
    def on_mc_remove(_):
        if S.curr_np is None: return set_status("Load an image first.")
        lbl = sam_label_dd.value
        _push_sam_undo()
        m = _binary_for_label(lbl)
        seg_masks[lbl] = _remove_small_regions_binary(m, mc_min_area.value).astype(np.uint8)
        _update_overlay()
        set_status(f"Removed small regions → {lbl} (min={mc_min_area.value})")
    mc_remove_btn.on_click(on_mc_remove)
    
    def _on_mask_lasso(verts):
        """Apply polygon selection drawn on RIGHT mask panel to current label using current apply mode."""
        if S.curr_np is None: 
            return set_status("Load an image first.")
        if last_color_mask_np is None:
            return set_status("Nothing to edit yet — create or load a mask first.")
        H, W = S.curr_np.shape[:2]
        # grid of pixel centers in display coords of ax_mask image
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        poly = MplPath(verts)
        sel = poly.contains_points(pts).reshape(H, W)
        if not np.any(sel):
            return set_status("Lasso selection empty.")
        _apply_mask_to_label(sel, sam_label_dd.value)
        set_status(f"Lasso → {sam_label_dd.value} ({sam_apply_mode.value})")
    
    def _toggle_mask_lasso(change):
        nonlocal mask_lasso
        if change["new"]:
            # activate
            if mask_lasso is not None:
                try: mask_lasso.disconnect_events()
                except: pass
                mask_lasso = None
            mask_lasso = LassoSelector(ax_mask, onselect=_on_mask_lasso)
            set_status("Mask lasso ON — draw polygon on right panel.")
        else:
            # deactivate
            if mask_lasso is not None:
                try: mask_lasso.disconnect_events()
                except: pass
            mask_lasso = None
            set_status("Mask lasso OFF.")
    mask_lasso_toggle.observe(_toggle_mask_lasso, names="value")
    
    # ===== Save overlay + color mask =====
    def on_sam_save(_):
        if S.curr_np is None or S.path is None:
            return set_status("Load an image first.")
        base_dir     = S.path.parent
        overlays_dir = base_dir / "overlays"
        masks_dir    = base_dir / "masks"
        overlays_dir.mkdir(exist_ok=True); masks_dir.mkdir(exist_ok=True)
    
        L, T, R, B = _current_bbox_for_filename()
        stem = S.path.stem
        overlay_name = f"{stem}_overlay_{L}_{T}_{R}_{B}.png"
        mask_name    = f"{stem}_mask_{L}_{T}_{R}_{B}.png"
    
        ov = _render_overlay_rgba(sam_alpha.value)
        base = S.curr_np.astype(np.float32).copy()
        if ov is not None and ov.shape[:2] == base.shape[:2]:
            a = (ov[..., 3:4].astype(np.float32)) / 255.0
            base = (1 - a) * base + a * ov[..., :3].astype(np.float32)
        overlay_out = overlays_dir / overlay_name
        cv2.imwrite(str(overlay_out), cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
        nonlocal last_color_mask_np
        color_mask = _render_color_mask_rgb()
        last_color_mask_np = color_mask.copy() if color_mask is not None else None
        if color_mask is None or not np.any(color_mask):
            return set_status(f"SAM: saved overlay → {overlay_out.name}; no mask content to save.")
    
        mask_out = masks_dir / mask_name
        cv2.imwrite(str(mask_out), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        set_status(f"SAM: saved overlay → {overlay_out.name}; mask → {mask_out.name}")
    sam_save_btn.on_click(on_sam_save)
    
    # ===== init =====
    refresh_lists()