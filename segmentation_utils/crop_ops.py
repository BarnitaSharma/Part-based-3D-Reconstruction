import numpy as np

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_crop(img, bbox, sym_on=False, sym_x=None):
    L, T, R, B = bbox
    H, W = img.shape[:2]

    L = clamp(L, 0, W)
    R = clamp(R, 0, W)
    T = clamp(T, 0, H)
    B = clamp(B, 0, H)

    if sym_on and sym_x is not None:
        half = max(sym_x - L, R - sym_x)
        L = clamp(sym_x - half, 0, W)
        R = clamp(sym_x + half, 0, W)

    if R <= L or B <= T:
        raise ValueError("Invalid crop")

    return img[T:B, L:R].copy(), (L, T, R, B)
