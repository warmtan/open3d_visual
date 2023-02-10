import open3d as o3d   #导入open3d库
import numpy as np    #导入numpy
 
 
points = np.random.rand(10000, 3)
point_cloud = o3d.PointCloud()
point_cloud.points = o3d.Vector3dVector(points)
o3d.draw_geometries([point_cloud])