import numpy as np
from utils.camera_geometry import *
import matplotlib.pyplot as plt

def project_colored_voxels(pts3d, colors, cam_pos, target, f, cx, cy, H, W):
    R = look_at_rotation(cam_pos, target)
    pts_cam = (pts3d - cam_pos) @ R.T

    X, Y, Z = pts_cam.T
    Z = np.where(Z < 1e-8, 1e-8, Z)

    u = (X / Z) * f + cx
    v = -(Y / Z) * f + cy

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    valid = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    proj_img = np.zeros((H, W, 3), dtype=np.uint8)
    proj_img[vi[valid], ui[valid]] = colors[valid]

    return proj_img


def visualize_reprojection(image, voxel_keypoints_dict, image_keypoints_dict, cam_params, title="Reprojection"):

    # Project voxel keypoints using final camera parameters
    projected = {}
    for name, pt3d in voxel_keypoints_dict.items():
        proj = project(pt3d, cam_params['cam_pos'], cam_params['target'],
                             cam_params['f'], cam_params['cx'], cam_params['cy'])
        projected[name] = tuple(proj)

    # Plot image with GT and projected keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for name in image_keypoints_dict:
        gt = image_keypoints_dict[name]
        pr = projected[name]

        # Ground truth in green
        plt.plot(gt[0], gt[1], 'go')
        plt.text(gt[0]+2, gt[1], f'{name}_GT', color='green', fontsize=9)

        # Projected in red
        plt.plot(pr[0], pr[1], 'ro')
        plt.text(pr[0]+2, pr[1], f'{name}_PR', color='red', fontsize=9)

    plt.title(title)
    plt.axis('off')
    plt.show()

    # === PRINTOUT COMPARISON ===
    print(f"\n{'Keypoint':<6} | {'GT (x, y)':<30} | {'Projected (x, y)':<30} | Error (L2)")
    print("-" * 80)
    total_err = 0
    for name in image_keypoints_dict:
        gt = np.array(image_keypoints_dict[name])
        pr = np.array(projected[name])
        err = np.linalg.norm(gt - pr)
        total_err += err
        print(f"{name:<6} | {tuple(np.round(gt, 2))!s:<30} | {tuple(np.round(pr, 2))!s:<30} | {err:.2f}")

    avg_err = total_err / len(image_keypoints_dict)
    print(f"\nAverage Reprojection Error: {avg_err:.2f} pixels")


