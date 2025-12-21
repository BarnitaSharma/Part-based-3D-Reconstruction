import numpy as np
import ipywidgets as W
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
from pathlib import Path

from .state import ImageState
from utils.config import PART_COLORS_NP


class SamUI:
    def __init__(self, *, state: ImageState, sam_predictor):
        self.S = state
        self.predictor = sam_predictor

        # ---------------- Parts & colors ----------------
        self.PART_COLORS = PART_COLORS_NP
        self.parts = list(self.PART_COLORS.keys())

        # ---------------- SAM state ----------------
        self.seg_masks = {}
        self.draw_order = []
        self.undo_stack = []

        # ---------------- Matplotlib ----------------
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.overlay_artist = None
        self.rs = None
        self.cid = None

        # ---------------- Widgets ----------------
        self.part_dd = W.Dropdown(
            options=self.parts,
            value=self.parts[0],
            description="Part:"
        )

        self.mode = W.ToggleButtons(
            options=[("Click", "click"), ("Box", "box")],
            value="click",
            description="SAM:"
        )

        self.alpha = W.FloatSlider(
            value=0.5, min=0.05, max=0.9, step=0.05,
            description="Alpha:"
        )

        self.undo_btn = W.Button(description="Undo")
        self.reset_btn = W.Button(description="Reset")
        self.save_btn = W.Button(description="Save")

        self.undo_btn.on_click(self._undo)
        self.reset_btn.on_click(self._reset)
        self.save_btn.on_click(self._save)

        self.mode.observe(self._on_mode_change, names="value")
        self.alpha.observe(lambda _: self._update_overlay(), names="value")

        display(
            W.VBox([
                W.HBox([self.part_dd, self.mode, self.alpha]),
                W.HBox([self.undo_btn, self.reset_btn, self.save_btn])
            ])
        )

    # ======================================================
    # Launch
    # ======================================================
    def launch(self):
        if self.S.curr_np is None:
            raise RuntimeError("state.curr_np is None")

        H, W = self.S.curr_np.shape[:2]

        self.seg_masks = {
            p: np.zeros((H, W), np.uint8) for p in self.parts
        }
        self.draw_order = self.parts.copy()
        self.undo_stack.clear()

        self.ax.clear()
        self.ax.imshow(self.S.curr_np)
        self.ax.set_axis_off()

        self.overlay_artist = self.ax.imshow(
            np.zeros((H, W, 4), np.uint8),
            interpolation="nearest",
            zorder=10
        )

        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )

        self._attach_box_selector()
        self._on_mode_change({"new": self.mode.value})

        plt.show()

    # ======================================================
    # Mode switching
    # ======================================================
    def _on_mode_change(self, change):
        if self.rs is None:
            return
        self.rs.set_active(change["new"] == "box")

    # ======================================================
    # Click / Box handling
    # ======================================================
    def _on_click(self, event):
        if self.mode.value != "click":
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        pos = 0 if event.key == "shift" else 1
        self._apply_sam_point(int(event.xdata), int(event.ydata), pos)

    def _attach_box_selector(self):
        if self.rs is not None:
            try:
                self.rs.disconnect_events()
            except Exception:
                pass

        self.rs = RectangleSelector(
            self.ax,
            self._on_box_select,
            interactive=False,
            useblit=False,
            minspanx=5,
            minspany=5,
            spancoords="pixels"
        )

    def _on_box_select(self, eclick, erelease):
        if self.mode.value != "box":
            return
        if eclick.xdata is None or erelease.xdata is None:
            return

        box = (
            int(min(eclick.xdata, erelease.xdata)),
            int(min(eclick.ydata, erelease.ydata)),
            int(max(eclick.xdata, erelease.xdata)),
            int(max(eclick.ydata, erelease.ydata)),
        )
        self._apply_sam_box(box)

    # ======================================================
    # SAM calls
    # ======================================================
    def _apply_sam_point(self, x, y, pos_label):
        self.predictor.set_image(self.S.curr_np)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[x, y]], np.float32),
            point_labels=np.array([pos_label], np.int32),
            multimask_output=True
        )
        best = masks[np.argmax(scores)]
        self._apply_mask(best.astype(bool))

    def _apply_sam_box(self, box):
        self.predictor.set_image(self.S.curr_np)
        masks, scores, _ = self.predictor.predict(
            box=np.array(box)[None, :],
            multimask_output=True
        )
        best = masks[np.argmax(scores)]
        self._apply_mask(best.astype(bool))

    # ======================================================
    # Mask logic
    # ======================================================
    def _apply_mask(self, mask_bool):
        part = self.part_dd.value
        self._push_undo()

        self.seg_masks[part][mask_bool] = 1

        if part in self.draw_order:
            self.draw_order.remove(part)
        self.draw_order.append(part)

        self._update_overlay()

    def _update_overlay(self):
        H, W = self.S.curr_np.shape[:2]
        ov = np.zeros((H, W, 4), np.uint8)
        a = int(self.alpha.value * 255)

        for part in self.draw_order:
            m = self.seg_masks[part]
            if not np.any(m):
                continue
            color = self.PART_COLORS[part]
            idx = m.astype(bool)
            ov[idx, :3] = color
            ov[idx, 3] = a

        self.overlay_artist.set_data(ov)
        self.fig.canvas.draw_idle()

    # ======================================================
    # Undo / Reset
    # ======================================================
    def _push_undo(self):
        self.undo_stack.append(
            {k: v.copy() for k, v in self.seg_masks.items()}
        )

    def _undo(self, _):
        if not self.undo_stack:
            return
        self.seg_masks = self.undo_stack.pop()
        self._update_overlay()

    def _reset(self, _):
        H, W = self.S.curr_np.shape[:2]
        self.seg_masks = {
            p: np.zeros((H, W), np.uint8) for p in self.parts
        }
        self.undo_stack.clear()
        self._update_overlay()

    # ======================================================
    # Save (FINAL, EXACT BEHAVIOR)
    # ======================================================
    def _save(self, _):
        if self.S.curr_np is None or self.S.path is None:
            print("Nothing to save")
            return

        # Source: data/<Monument>/images/<image>.jpg
        img_path = self.S.path
        monument = img_path.parents[1].name
        image_stem = img_path.stem

        # Target: data_temp/<Monument>/masks/<image>.png
        project_root = img_path.parents[2].parent
        data_temp = project_root / "data_temp"
        masks_dir = data_temp / monument / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        H, W = self.S.curr_np.shape[:2]
        color_mask = np.zeros((H, W, 3), np.uint8)

        for part in self.draw_order:
            m = self.seg_masks[part]
            if not np.any(m):
                continue
            color = self.PART_COLORS[part]
            color_mask[m.astype(bool)] = color

        out_path = masks_dir / f"{image_stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

        print(f"Saved mask → {out_path}")
