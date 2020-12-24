import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy
import math
def pc_colors(arr):
    list = np.asarray([
        # [100, 100, 100],
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors)/255

def show_range(points,label,window_name='test'):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    rail_index = label == 1
    rail_points = points[rail_index]
    x_poly = rail_points[:, 2]
    x_poly = x_poly[:, np.newaxis]
    y_poly = rail_points[:, 1]
    y_poly = y_poly[:, np.newaxis]

    # degree=2 y =         c*x^2 + b*x + a
    # degree=3 y = d*x^3 + c*x^2 + b*x + a
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x_poly)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_poly)
    poly_para = [lin_reg.intercept_, lin_reg.coef_]
    print('poly_para : ',poly_para)

    # points:右下[h,x+3,y]  左下[h,x-3,y]  右上[h+5,x+3,y]  左上[h+5,x-3,y]
    # lines:L=rail_points.shape[0]
    #       [0,1] [1,2]...[L-2,L-1]  右下
    #       L+ [0,1] [1,2]...[L-2,L-1] 左下
    #       2L+ [0,1] [1,2]...[L-2,L-1] 右上
    #       3L+ [0,1] [1,2]...[L-2,L-1] 左上
    #       [0,L] [L,2L] [2L,3L] [3L,0] 首框
    #       [L-1,2L-1] [2L-1,3L-1] [3L-1,4L-1] [4L-1,L-1] 尾框

    min_x = rail_points[:,2].min()
    x = np.arange(min_x,70+min_x,1).reshape(-1,1)
    n_sample=(x).shape[0]

    y = poly_para[0]+poly_para[1][0][0]*x + poly_para[1][0][1]*x**2 #+ poly_para[1][0][2]*x**3
    h = rail_points[:,0].max()

    line_points_rd = np.concatenate((h*np.ones([n_sample,1]),y+1.7,x),axis=1)
    line_points_ld = np.concatenate((h*np.ones([n_sample,1]),y-1.7,x),axis=1)
    line_points_rt = np.concatenate((h * np.ones([n_sample, 1])+4, y + 1.7, x), axis=1)
    line_points_lt = np.concatenate((h * np.ones([n_sample, 1])+4, y - 1.7, x), axis=1)
    line_points = np.concatenate((line_points_rd,line_points_ld,
                                  line_points_rt,line_points_lt),axis=0)

    line_lines = [[j+i*n_sample,j+1+i*n_sample] for i in range(4) for j in range(n_sample-1)]

    line_lines = np.asarray(line_lines)
    line_lines_add = np.asarray([[0,n_sample],[n_sample,3*n_sample],
                                 [2*n_sample,3*n_sample],
                                 [0*n_sample,2*n_sample],
                                 [n_sample-1,2*n_sample-1],
                                 [2*n_sample-1,4*n_sample-1],
                                 [3*n_sample-1,4*n_sample-1],
                                 [3*n_sample-1,n_sample-1]
                                 ])

    line_lines = np.concatenate((line_lines,line_lines_add),axis=0)

    line_color = [[0, 0, 0] for i in range(len(line_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_color)

    colors = pc_colors(label.squeeze())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(colors.squeeze())
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(2, center=mesh.get_center())
    o3d.visualization.draw_geometries([mesh,pcd],window_name=window_name,
                                      width=1080, height=1080)
def check_label():
    data_root = '/media/hwq/g/qxdpcdascii/labeled_rail/pc_npy/a3'

    files = os.listdir(data_root)
    files.sort()
    print(len(files))
    # 下上 左右 前后  X Y Z Z与Y的关系  Y=f(Z)
    for i, file in enumerate(files):
        if i >= 44:
            data = np.load(os.path.join(data_root,file))
            print(os.path.join(data_root,file),data.shape)
            points = data[:,0:3]
            label = data[:,4].astype(np.int32)
            print('points',label.max())
            show_range(points,label,window_name=str(i))


if __name__ == '__main__':
    check_label()
