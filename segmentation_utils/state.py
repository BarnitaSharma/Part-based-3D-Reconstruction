# mask_generation_utils/state.py
from pathlib import Path
from typing import Optional
import numpy as np

class ImageState:
    def __init__(self):
        self.path: Optional[Path] = None
        self.base_np: Optional[np.ndarray] = None
        self.curr_np: Optional[np.ndarray] = None

        self.sym_on: bool = False
        self.sym_x: Optional[int] = None

        self.undo = []
