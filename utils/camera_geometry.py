import numpy as np

def look_at_rotation(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
    z = target - eye
    z /= np.linalg.norm(z)

    if np.allclose(np.abs(np.dot(z, up)), 1.0):
        up = np.array([0, 0, 1], dtype=np.float32)

    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    return np.stack([x, y, z], axis=0)


def project(pt3d, cam_pos, target, f, cx, cy):
    R = look_at_rotation(cam_pos, target)
    pt_cam = (pt3d - cam_pos) @ R.T

    X, Y, Z = pt_cam
    Z = max(Z, 1e-8)

    u = (X / Z) * f + cx
    v = -(Y / Z) * f + cy

    return np.array([u, v])
