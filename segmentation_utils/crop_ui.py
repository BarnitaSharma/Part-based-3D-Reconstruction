from pathlib import Path
from PIL import Image
import numpy as np
import ipywidgets as W
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from .state import ImageState


class CropUI:
    def __init__(self, *, state: ImageState, save_dir: Path):
        self.S = state
        self.save_dir = save_dir

        # ---------------- UI widgets ----------------
        self.sym_toggle = W.ToggleButtons(
            options=[("Sym ON", True), ("Sym OFF", False)],
            value=False,
            description="Sym:"
        )

        self.sym_auto = W.Button(description="Auto axis")
        self.sym_x_box = W.BoundedIntText(
            value=0, min=0, max=99999, description="Axis x:"
        )

        self.crop_btn = W.Button(description="Crop", button_style="success")
        self.undo_btn = W.Button(description="Undo")
        self.reset_btn = W.Button(description="Reset")
        self.save_btn = W.Button(description="Save")

        self.status = W.HTML("<code>Ready.</code>")

        self.sym_toggle.observe(self._on_sym_toggle, names="value")
        self.sym_auto.on_click(self._on_sym_auto)
        self.sym_x_box.observe(self._on_sym_x, names="value")

        self.crop_btn.on_click(self._crop)
        self.undo_btn.on_click(self._undo)
        self.reset_btn.on_click(self._reset)
        self.save_btn.on_click(self._save)

        display(W.VBox([
            W.HBox([self.sym_toggle, self.sym_auto, self.sym_x_box]),
            W.HBox([self.crop_btn, self.undo_btn, self.reset_btn, self.save_btn]),
            self.status
        ]))

        # ---------------- matplotlib ----------------
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()
        self.rs = None
        self.sym_line = None

    # ======================================================
    # Load image
    # ======================================================
    def load(self, image_path: Path):
        self.S.path = image_path
        self.S.base_np = np.array(Image.open(image_path).convert("RGB"))
        self.S.curr_np = self.S.base_np.copy()
        self.S.undo = []

        self.ax.clear()
        self.ax.imshow(self.S.curr_np)
        self.ax.set_axis_off()

        self._ensure_selector()

        # ---- symmetry init ----
        self.S.sym_x = self._find_symmetry_axis(self.S.curr_np)
        self.sym_x_box.max = self.S.curr_np.shape[1]
        self.sym_x_box.value = int(self.S.sym_x)
        self._draw_sym_line()

        self._set_status(f"Loaded {image_path.name}")
        plt.show()

    # ======================================================
    # Symmetry helpers
    # ======================================================
    def _find_symmetry_axis(self, img):
        H, W = img.shape[:2]
        gray = img.mean(axis=2)
        best_x = W // 2
        best_err = float("inf")

        for x in range(W // 4, 3 * W // 4):
            left = gray[:, :x][:, ::-1]
            right = gray[:, x:]
            w = min(left.shape[1], right.shape[1])
            if w <= 0:
                continue
            err = np.abs(left[:, :w] - right[:, :w]).mean()
            if err < best_err:
                best_err = err
                best_x = x
        return int(best_x)

    def _draw_sym_line(self):
        if not self.sym_toggle.value:
            if self.sym_line:
                self.sym_line.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        if self.sym_line is None:
            self.sym_line = self.ax.axvline(
                self.S.sym_x, color="red", linestyle="--"
            )
        else:
            self.sym_line.set_xdata([self.S.sym_x, self.S.sym_x])
            self.sym_line.set_visible(True)

        self.fig.canvas.draw_idle()

    def _on_sym_toggle(self, change):
        self._draw_sym_line()

    def _on_sym_auto(self, _):
        self.S.sym_x = self._find_symmetry_axis(self.S.curr_np)
        self.sym_x_box.max = self.S.curr_np.shape[1]
        self.sym_x_box.value = int(self.S.sym_x)
        self._draw_sym_line()
        self._set_status(f"Auto symmetry x={self.S.sym_x}")

    def _on_sym_x(self, change):
        self.S.sym_x = int(change["new"])
        self._draw_sym_line()

    # ======================================================
    # Crop logic
    # ======================================================
    def _ensure_selector(self):
        if self.rs is not None:
            try:
                self.rs.disconnect_events()
            except Exception:
                pass

        self.rs = RectangleSelector(
            self.ax,
            lambda *_: None,
            interactive=True,
            minspanx=5,
            minspany=5,
            spancoords="pixels"
        )

    def _crop(self, _):
        if self.rs is None:
            return

        x0, x1, y0, y1 = self.rs.extents
        L, R = int(min(x0, x1)), int(max(x0, x1))
        T, B = int(min(y0, y1)), int(max(y0, y1))

        if R <= L or B <= T:
            self._set_status("Invalid crop.")
            return

        H, W = self.S.curr_np.shape[:2]

        # ---- symmetric crop ----
        if self.sym_toggle.value:
            half = max(self.S.sym_x - L, R - self.S.sym_x)
            L = max(0, self.S.sym_x - half)
            R = min(W, self.S.sym_x + half)

        self.S.undo.append(self.S.curr_np.copy())
        self.S.curr_np = self.S.curr_np[T:B, L:R]

        # ---- recompute symmetry AFTER crop (IMPORTANT FIX) ----
        self.S.sym_x = self._find_symmetry_axis(self.S.curr_np)
        self.sym_x_box.max = self.S.curr_np.shape[1]
        self.sym_x_box.value = int(self.S.sym_x)

        self.ax.clear()
        self.ax.imshow(self.S.curr_np)
        self.ax.set_axis_off()
        self._ensure_selector()
        self._draw_sym_line()

        self._set_status(f"Cropped → {self.S.curr_np.shape}")

    def _undo(self, _):
        if not self.S.undo:
            self._set_status("Nothing to undo.")
            return

        self.S.curr_np = self.S.undo.pop()

        self.ax.clear()
        self.ax.imshow(self.S.curr_np)
        self.ax.set_axis_off()
        self._ensure_selector()

        self.S.sym_x = self._find_symmetry_axis(self.S.curr_np)
        self.sym_x_box.max = self.S.curr_np.shape[1]
        self.sym_x_box.value = int(self.S.sym_x)
        self._draw_sym_line()

        self._set_status("Undo.")

    def _reset(self, _):
        self.S.curr_np = self.S.base_np.copy()
        self.S.undo.clear()

        self.ax.clear()
        self.ax.imshow(self.S.curr_np)
        self.ax.set_axis_off()
        self._ensure_selector()

        self.S.sym_x = self._find_symmetry_axis(self.S.curr_np)
        self.sym_x_box.max = self.S.curr_np.shape[1]
        self.sym_x_box.value = int(self.S.sym_x)
        self._draw_sym_line()

        self._set_status("Reset.")

    # ======================================================
    # Save
    # ======================================================
    def _save(self, _):
        if self.S.curr_np is None or self.S.path is None:
            self._set_status("Nothing to save.")
            return

        img_path = self.S.path
        monument = img_path.parents[1].name
        filename = img_path.name

        out_dir = self.save_dir / monument / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / filename
        Image.fromarray(self.S.curr_np).save(out_path)

        self._set_status(f"Saved → {out_path}")

    # ======================================================
    def _set_status(self, msg):
        self.status.value = f"<code>{msg}</code>"
