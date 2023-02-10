import open3d as o3d
import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation

def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

# left
# 0.555818	-0.824037	-0.10968	708.485
# -0.806888	-0.566519	0.167295	-286.396
# -0.199993	-0.00448642	-0.979787	1604.1
# 0	0	0	1

# right
# -0.805804	-0.563352	0.182519	293.528
# -0.552033	0.826158	0.112794	-304.755
# -0.214333	-0.00986658	-0.976711	1611.44
# 0	0	0	1
left_transform = np.array([
    [0.555818,-0.824037,-0.10968,708.485],
    [-0.806888,-0.566519,0.167295,-286.396],
    [-0.199993,-0.00448642,-0.979787,1604.1],
    [0., 0., 0., 1.]])
left_transform[:3, -1] = left_transform[:3, -1] / 1000
right_transform = np.array([
    [-0.805804,-0.563352,0.182519,293.528],
    [-0.552033,0.826158,0.112794,-304.755],
    [-0.214333,-0.00986658,-0.976711,1611.44],
    [0., 0., 0., 1.]])
right_transform[:3, -1] = right_transform[:3, -1] / 1000
print(("--left 求逆--"))
left_to_camera_transform = np.linalg.inv(left_transform)
print(("--right 求逆--"))
right_to_camera_transform = np.linalg.inv(right_transform)


left_z_dir = (left_to_camera_transform[0:3,0:3] @ np.array([0., 0., 1.])[:, np.newaxis])[:, 0]
right_z_dir = (right_to_camera_transform[0:3,0:3] @ np.array([0., 0., 1.])[:, np.newaxis])[:, 0]

camera_origin = np.array([0., 0., 0.])
camera_x_dir = np.array([1., 0., 0.])
camera_y_dir = np.array([0., 1., 0.])
camera_z_dir = np.array([0., 0., 1.])

world_z_dir = (left_z_dir + right_z_dir) / 2
world_z_dir = world_z_dir / np.linalg.norm(world_z_dir)
left_origin = left_to_camera_transform[:3, -1]
right_origin = right_to_camera_transform[:3, -1]
world_origin = (left_origin + right_origin) / 2
world_x_dir = (right_origin - left_origin) / np.linalg.norm(right_origin - left_origin)

world_sample_pts = np.stack([world_origin, 
                             world_origin + world_x_dir,
                             world_origin + 0.5 * world_x_dir,
                             world_origin + world_z_dir,
                             world_origin + 0.5 * world_z_dir])
camera_sample_pts = np.stack([camera_origin, 
                              camera_origin + camera_x_dir,
                              camera_origin + 0.5 * camera_x_dir,
                              camera_origin + camera_z_dir,
                              camera_origin + 0.5 * camera_z_dir])
world_sample_pts_pcd = o3d.geometry.PointCloud()
world_sample_pts_pcd.points = o3d.utility.Vector3dVector(world_sample_pts)
camera_sample_pts_pcd = o3d.geometry.PointCloud()
camera_sample_pts_pcd.points = o3d.utility.Vector3dVector(camera_sample_pts)

c, R, t = umeyama(camera_sample_pts.T, world_sample_pts.T)
print(f"c: {c}, R: {R}, t: {t}")
world_to_camera_transform = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0., 0., 0., 1.]])], axis=0)
print(world_to_camera_transform)

world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(world_to_camera_transform)
camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])
left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(left_to_camera_transform)
right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(right_to_camera_transform)
o3d.visualization.draw_geometries([camera, left, right, world])  # camera coord system

world_to_left_transform = np.linalg.inv(left_to_camera_transform) @ world_to_camera_transform
world_to_right_trasnform = np.linalg.inv(right_to_camera_transform) @ world_to_camera_transform
world_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
left_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(np.linalg.inv(world_to_left_transform))
right_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(np.linalg.inv(world_to_right_trasnform))
# o3d.visualization.draw_geometries([world_in_world, left_in_world, right_in_world])  # world coord system

left_rpy_in_world = Rotation.from_matrix(np.linalg.inv(world_to_left_transform)[:3, :3]).as_euler('xyz')
right_rpy_in_world = Rotation.from_matrix(np.linalg.inv(world_to_right_trasnform)[:3, :3]).as_euler('xyz')
print(f'left_rpy_in_world: {left_rpy_in_world}')
print(f'right_rpy_in_world: {right_rpy_in_world}')