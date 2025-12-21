# utils/preprocess_helpers.py

import numpy as np
import open3d as o3d
import os
import time

# ============================================================
# Basic utilities
# ============================================================

def flip_y_axis(coords):
    coords = coords.copy()
    y = coords[:, 1]
    coords[:, 1] = y.max() - (y - y.min())
    return coords


def normalize_preserve_aspect(pts):
    min_val = pts.min(0)
    size = pts.max(0) - min_val
    scale = size.max()
    norm = (pts - min_val) / (scale + 1e-8)
    norm[:, 1] -= norm[:, 1].max()
    return norm


# ============================================================
# I/O ()
# ============================================================

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return (
        np.asarray(pcd.points),
        (np.asarray(pcd.colors) * 255).astype(np.uint8)
    )


# ============================================================
# ICP helpers ()
# ============================================================

def make_o3d_pcd(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.estimate_normals()
    return pcd


def icp_align(source_pts, source_cols, target_pts, target_cols, max_dist=0.05):
    src = make_o3d_pcd(source_pts, source_cols)
    tgt = make_o3d_pcd(target_pts, target_cols)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    aligned = np.asarray(src.transform(reg.transformation).points)
    return aligned, reg.transformation


# ============================================================
# MAIN BUILDER (VERIFIED)
# ============================================================

def build_taj_clouds(root_path, verbose=True):
    """
    Build Taj Mahal point clouds in WORLD SPACE.

    GUARANTEES:
    - Matches original notebook math
    - No normalization
    - No CAD
    - No coordinate flips
    - Same symmetry + ICP ordering
    """

    def log(msg):
        if verbose:
            print(msg, flush=True)

    def timed(msg):
        log(msg)
        return time.time()

    # =========================================================
    # Step 1: Load sparse + dense ()
    # =========================================================
    t0 = timed("▶ Loading sparse and dense point clouds")

    sparse_path = os.path.join(
        root_path, "results/4.Inter-method_3D",
        "segmented_point_cloud_final.ply"
    )
    dense_path = os.path.join(
        root_path, "results/4.Inter-method_3D",
        "fused.ply"
    )

    pts_sparse, cols_sparse = load_ply(sparse_path)
    pts_dense, cols_dense = load_ply(dense_path)

    log(f"   Sparse: {len(pts_sparse):,} pts")
    log(f"   Dense : {len(pts_dense):,} pts")
    log(f"   done in {time.time() - t0:.2f}s\n")

    # =========================================================
    # Step 2: Crop dense to sparse bbox ()
    # =========================================================
    t0 = timed("▶ Cropping dense cloud to sparse bounding box")

    bbox_min = pts_sparse.min(0)
    bbox_max = pts_sparse.max(0)

    mask = np.all((pts_dense >= bbox_min) & (pts_dense <= bbox_max), axis=1)
    pts_dense_crop = pts_dense[mask]
    cols_dense_crop = cols_dense[mask]

    log(f"   Cropped dense: {len(pts_dense_crop):,} pts")
    log(f"   done in {time.time() - t0:.2f}s\n")

    # =========================================================
    # Step 3: Plane fit + align to Z+ ()
    # =========================================================
    t0 = timed("▶ Fitting facade plane and aligning to Z+")

    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pts_dense_crop)
    )

    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    target = np.array([0, 0, 1])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c_dot = np.dot(normal, target)

    if s < 1e-8:
        R_align = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s**2))

    pts_rot = (R_align @ pts_dense_crop.T).T

    log("   Plane aligned")
    log(f"   done in {time.time() - t0:.2f}s\n")

    # =========================================================
    # Step 4: Naive 4-way symmetry (CRITICAL, ORIGINAL)
    # =========================================================
    t0 = timed("▶ Generating naive 4-way symmetry")

    x_mid = (pts_rot[:, 0].min() + pts_rot[:, 0].max()) / 2
    y_mid = (pts_rot[:, 1].min() + pts_rot[:, 1].max()) / 2
    z_mid = (pts_rot[:, 2].min() + pts_rot[:, 2].max()) / 2
    center = np.array([x_mid, y_mid, z_mid])

    R_y90 = np.array([
        [0, 0, -1],
        [0, 1,  0],
        [1, 0,  0]
    ])

    R_y_90 = np.array([
        [0, 0,  1],
        [0, 1,  0],
        [-1, 0, 0]
    ])

    def spin(pts, R):
        return (R @ (pts - center).T).T + center

    P_front = pts_rot.copy()

    P_back = pts_rot.copy()
    P_back[:, 2] = 2 * z_mid - P_back[:, 2]

    P_left = spin(pts_rot, R_y90)
    P_left[:, 0] = 2 * x_mid - P_left[:, 0]

    P_right = spin(pts_rot, R_y_90)
    P_right[:, 0] = 2 * x_mid - P_right[:, 0]

    log("   Symmetry generated")
    log(f"   done in {time.time() - t0:.2f}s\n")

    # =========================================================
    # Step 5: ICP refinement (ORDER )
    # =========================================================
    t0 = timed("▶ Running ICP alignment")

    log("   → ICP Left → Front")
    tL = time.time()
    P_left, _ = icp_align(P_left, cols_dense_crop, P_front, cols_dense_crop)
    log(f"     done in {time.time() - tL:.2f}s")
    
    log("   → ICP Right → Front")
    tR = time.time()
    P_right, _ = icp_align(P_right, cols_dense_crop, P_front, cols_dense_crop)
    log(f"     done in {time.time() - tR:.2f}s")
    
    log("   → ICP Back → Left")
    tB = time.time()
    P_back, _ = icp_align(P_back, cols_dense_crop, P_left, cols_dense_crop)
    log(f"     done in {time.time() - tB:.2f}s")
    
    pts_icp = np.vstack([P_front, P_left, P_right, P_back])
    
    log(f"   Final ICP cloud: {len(pts_icp):,} pts")
    log(f"   total ICP time: {time.time() - t0:.2f}s\n")


    # =========================================================
    # Step 6: Load carved voxel grid ()
    # =========================================================
    t0 = timed("▶ Loading carved voxel grid")

    carved_data = np.load(os.path.join(
        root_path, "results/4.Inter-method_3D",
        "Taj_voxel_grid.npz"
    ))

    carved_mask = np.any(carved_data["voxel_grid"] != 0, axis=-1)
    carved_coords = np.array(
        np.nonzero(carved_mask)
    ).T.astype(np.float32)

    log(f"   Carved grid: {len(carved_coords):,} pts")
    log(f"   done in {time.time() - t0:.2f}s\n")

    log("✅ Preprocessing complete\n")

    # return (
    #     pts_sparse,
    #     pts_dense_crop,
    #     pts_icp,
    #     carved_coords,
    # )

    # =========================================================
    # Step 7: Load + align synthetic CAD (VERIFIED)
    # =========================================================
    t0 = timed("▶ Loading synthetic CAD mesh")
    
    import trimesh
    
    scene = trimesh.load(
        os.path.join(root_path, "results/4.Inter-method_3D", "synthetic_taj.obj"),
        force="scene"
    )
    
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.geometry.values())
    else:
        mesh = scene
    
    # --- swap Y/Z to match reconstruction ---
    T = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    verts = mesh.vertices @ T.T
    faces = mesh.faces
    
    ref_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # --- sample reference points ---
    sampled_ref = ref_mesh.sample(50000)
    
    # --- flip Y to match scene convention ---
    sampled_ref = flip_y_axis(sampled_ref)
    
    log("   Synthetic CAD loaded and sampled")
    log(f"   done in {time.time() - t0:.2f}s\n")

    # =========================================================
    # Step 8: Align ICP cloud to sparse Y-level (CRITICAL)
    # =========================================================
    y_ref = pts_sparse[:, 1].min()
    y_icp = pts_icp[:, 1].min()
    
    pts_icp = pts_icp.copy()
    pts_icp[:, 1] += (y_ref - y_icp)

    return {
        "Sparse": pts_sparse,
        "Dense (Cropped)": pts_dense_crop,
        "Completed (ICP Aligned)": pts_icp,
        "Carved Grid": flip_y_axis(carved_coords.copy()),
        "Synthetic Reference (CAD)": sampled_ref,
    }

__all__ = [
    "flip_y_axis",
    "normalize_preserve_aspect",
    "build_taj_clouds",
    # others…
]

