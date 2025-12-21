from pathlib import Path
import numpy as np

ROOT_PATH = Path.cwd() / "data"

MONUMENT_CONFIG = {
    "Akbar": {
        "front": ["_front_mask.png"],
        "drone": "_drone_mask.png",
    },
    "Bibi": {
        "front": ["_front_mask.png"],
        "drone": "_drone_mask.png",
    },
    "Charminar": {
        "front": ["_front_mask.png", "_front_mask_win.png"],
        "drone": "_drone_mask.png",
    },
    "Itimad": {
        "front": ["_front_mask.png"],
        "drone": "_drone_mask.png",
    },
    "Taj": {
        "front": ["_front_mask.png"],
        "drone": "_drone_mask.png",
    },
}

PART_COLORS = {
    "full_building": (253, 248, 96),
    "chhatris": (1, 220, 5),
    "plinth": (63, 138, 173),
    "dome": (190, 0, 255),
    "front_minarets": (0, 0, 255),
    "back_minarets": (5, 223, 223),
    "small_minarets": (255, 180, 80),
    "main_door": (180, 140, 255),
    "windows": (255, 120, 230),
    "background": (216, 224, 251),
}

PART_COLORS_NP = {k: np.array(v) for k, v in PART_COLORS.items()}
INTERIOR_PARTS = ["main_door", "windows"]

MAX_DIM = 256
