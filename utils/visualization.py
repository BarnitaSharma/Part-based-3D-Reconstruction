import numpy as np
import plotly.graph_objects as go
import trimesh

def visualize_trimesh(verts,faces,normals):
    # occ = np.any(colored > 0, axis=-1)
    # verts, faces, normals, _ = marching_cubes(occ.astype(float), 0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.show()

def visualize_mesh_plotly(verts, faces, vertex_colors=None, title="Colored Voxel Mesh"):
    """
    Visualize a mesh with optional vertex colors using Plotly.
    """

    if vertex_colors is not None and vertex_colors.max() > 1:
        vertex_colors = vertex_colors / 255.0

    fig = go.Figure(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_colors,
        flatshading=True,
        opacity=1,
        lighting=dict(ambient=1),
    ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        title=title
    )
    fig.show()

def plot_voxel(points, colors=None,title='3D Visualization'):
    if colors is None:
        color_input = 'blue'
    else:
        colors = np.asarray(colors)
        if colors.ndim == 2 and colors.shape[1] == 3 and colors.max() > 1:
            colors = colors / 255
        color_input = colors

    fig = go.Figure(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color_input, opacity=1)
    ))
    fig.update_layout(scene=dict(aspectmode='data'), title=title)
    fig.show()
