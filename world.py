import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
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
left = np.array([
    [0.555818,-0.824037,-0.10968,708.485],
    [-0.806888,-0.566519,0.167295,-286.396],
    [-0.199993,-0.00448642,-0.979787,1604.1],
    [0., 0., 0., 1.]])
left[:3, -1] = left[:3, -1] / 1000
right = np.array([
    [-0.805804,-0.563352,0.182519,293.528],
    [-0.552033,0.826158,0.112794,-304.755],
    [-0.214333,-0.00986658,-0.976711,1611.44],
    [0., 0., 0., 1.]])
right[:3, -1] = right[:3, -1] / 1000
print(("--left 求逆--"))
print(np.linalg.inv(left))
print(("--right 求逆--"))
print(np.linalg.inv(right))
r_l = R.from_matrix(left[0:3,0:3])
euler_l = r_l.as_euler('xyz', degrees=True)
print(("-euler_l-"))
print(euler_l)

r_r = R.from_matrix(right[0:3,0:3])
euler_r = r_r.as_euler('xyz', degrees=True)
print(("-euler_r-"))
print(euler_r)

euler_w = [(euler_l[0]+euler_r[0])/2,(euler_l[1]+euler_r[1])/2,(euler_l[2]+euler_r[2])/2]
euler_w = np.array(euler_w)
print("--世界的角度--")
print(euler_w)
# 欧拉角到旋转矩阵
r4 = R.from_euler('xyz', euler_w, degrees=True)
rm = r4.as_matrix()
# print(rm)

world_old = [(left[0][3]+right[0][3])/2,
(left[1][3]+right[1][3])/2,
(left[2][3]+right[2][3])/2]
world_old = np.array(world_old)
print("--world_old--")
print(world_old)
# print(np.linalg.inv(left))
# print(np.linalg.inv(right))
world_origin = [(np.linalg.inv(left)[0][3]+np.linalg.inv(right)[0][3])/2,
(np.linalg.inv(left)[1][3]+np.linalg.inv(right)[1][3])/2,
(np.linalg.inv(left)[2][3]+np.linalg.inv(right)[2][3])/2]
world_origin = np.array(world_origin)
print(("--世界原点 求逆--"))
print(world_origin)

world_size = [[rm[0][0],rm[0][1],rm[0][2],world_origin[0]],
       [rm[1][0],rm[1][1],rm[1][2],world_origin[1]],
       [rm[2][0],rm[2][1],rm[0][2],world_origin[2]],
       [0,0,0,1]]
world_size = np.array(world_size)
print(("--世界坐标系--"))
print(world_size)


# camera = o3d.geometry.create_mesh_coordinate_frame(size=2,origin=[0,0,0])
# left = o3d.geometry.create_mesh_coordinate_frame(size=1).transform(np.linalg.inv(left))
# right = o3d.geometry.create_mesh_coordinate_frame(size=1).transform(np.linalg.inv(right))
# # world = o3d.geometry.create_mesh_coordinate_frame(size=1.5,origin=[world_origin[0],world_origin[1],world_origin[2]])
# world = o3d.geometry.create_mesh_coordinate_frame(size=1).transform(world_size)
# o3d.visualization.draw_geometries([camera]+[left]+[right]+[world])