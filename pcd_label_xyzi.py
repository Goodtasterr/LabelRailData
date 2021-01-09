import open3d as o3d
import numpy as np
import time
import os
import copy
from file_prepare_rail import pcd2xyzi
from plt_line import label_points_lr

def pc_range(points,range):
    '''

    :param points: nparray:[N,4]
    :param range: list:[6] x_min,x_max,y_min,y_max,z_min,z_max
    :return: [N',4]
    '''
    index_x = (points[:, 0] > (range[0])) & (points[:, 0] < range[1])
    index_y = (points[:, 1] > (range[2])) & (points[:, 1] < range[3])
    index_z = (points[:, 2] > (range[4])) & (points[:, 2] < range[5])
    index = index_x & index_y & index_z
    points_ranged = points[index]

    return points_ranged

def pc_colors(arr):
    list = np.asarray([
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
def o3d_paint(points,color=False,name='none'): #points: numpy array [N,4] or [N,5] with label
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    if color:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors(points[:,-1]))
    else:
        color_array = np.concatenate((np.zeros((points.shape[0],2))+0.7,points[:, -1,np.newaxis]/255),axis=1)
        pcd.colors = o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([pcd],window_name=name,width=800, height=600)


def label_points(points,labels,parameter,ranges):
    '''
    :param points: nparray:[N,3]
    :param labels: nparray:[n]:classes number
    :param parameter: nparray:[n]:a,_,b,dist1,dist2,..., shape is same as number
    :param range: list:[n,6] n:classees; 6:x_min,x_max,y_min,y_max,z_min,z_max
    :return: [N',3]
    '''
    # global bbb
    label = np.zeros([points.shape[0]]).astype(int)
    part_index = []
    for i,range in enumerate(ranges):
        ranged = copy.deepcopy(range)
        if i ==0:
            idx_1 = (points[:,2]<parameter[3])
            idx_2 = (points[:,2]<parameter[4]) &(points[:,2]>parameter[3])
            idx_3 = (points[:,2]>parameter[4])
            ranged[3] +=  parameter[2]*points[:,2]*idx_3
            ranged[3] += (parameter[0]*((points[:,2]-parameter[4])**2)+parameter[2]*points[:,2])*idx_2
            ranged[3] += ((2*parameter[0]*(parameter[3]-parameter[4])+parameter[2])*(points[:,2]-parameter[3])
                          +parameter[0]*((parameter[3]-parameter[4])**2)+parameter[2]*parameter[3])*idx_1

            quad_idx = (points[:,2]<parameter[4]) &(points[:,2]>parameter[3])
            # linear_idx = (points[:,2]<parameter[3])
            # ranged[3] = range[3]+parameter[0]*(
            #         (points[:,2]-parameter[4])**2)*quad_idx + parameter[2]*points[:,2] + parameter[2]*(points[:,2]-parameter[3])*linear_idx
            ranged[2] = ranged[2] + ranged[3] - (1.5*(((parameter[0]*points[:,2]*2*quad_idx+parameter[2])**2+1)**(1/2)))
        else:
            ranged[3] = range[3] - parameter[2] * (points[:, 2])
            ranged[2] = range[2] - parameter[2] * (points[:, 2])
        index_x = (points[:, 0] > (ranged[0])) & (points[:, 0] < ranged[1])
        index_y = (points[:, 1] > (ranged[2])) & (points[:, 1] < ranged[3])
        index_z = (points[:, 2] > (ranged[4])) & (points[:, 2] < ranged[5])
        index = index_x & index_y & index_z
        part_index.append(index)
        index2 = np.asarray(index).astype(int)
        index2 = index2*labels[i]
        label+=(index2)
    return label, part_index



if __name__ == '__main__':

    root = '../../dataset/qxdpcdascii/'
    train_index_root = os.path.join(root,'train_index')
    train_file_root = 'a3'  # pcd files in a1
    save_npy_root = os.path.join(root,'labeled_rail','pc_npy',train_file_root).replace('\\', '/')
    save_para_root = os.path.join(root,'labeled_rail','parameter',train_file_root).replace('\\', '/')
    os.makedirs(save_npy_root) if not os.path.exists(save_npy_root) else None
    os.makedirs(save_para_root) if not os.path.exists(save_para_root) else None

    files_name = np.loadtxt(os.path.join(train_index_root,train_file_root+'.txt').replace('\\', '/'),dtype=str)

    rail_range = [-5.5, 7, -30, 30, 6.5, 70] #去掉范围外的离散点
    trans_xyz = [[0,1,0],
                 [1,0,0],
                 [0,0,-1]]
    speed = 1.8
    #多目标 位置区间　第一行为轨道，之后都是电线杆
    rail_ranges = [[-5.5, -2.4, 0, 1, 6, 56],  #轨道

              [ 1.2,   6.5, -2.4,   4.9, 6,   58],  # 电线杆上

              [-2.7, 1.2, 2., 5, 22, 31],  # 电线杆１
              [-2.7, 1.2,0, 5, 38, 58],  # 电线杆左前
              [-2.6,  6.5,    -3,    3.5,  62, 62],  # 电线杆２

              [-2.8,  1.2,    -6.8,    -3,  42, 42],  # 电线杆3
              ]
    labels = [1,2,2,2,2,2] #label parameter 1 : label value

    # [a,_,b,_,_]  直道是一次函数n=1，弯道是二次函数n>1
    #shape of number is same as number2 and rail_range.shape[1]
    # parameter = [-0.006,0,-9.6/70,15,22,0] #label parameter 2 : y=a*z^2 + b*z
    parameter_lr = np.asarray([0.06,-0.76,-0.101,3.6,170,1.65]).astype(np.float32)  #parameter_lr k1 b1 k2 b2 r delta_r
    for i, file_name in enumerate(files_name):
        if i>=352:
            print('Labeling NO.%d file: %s...in part %s with %d files... '%(i,file_name,train_file_root,len(files_name)))
            points = pcd2xyzi(os.path.join(root,train_file_root,file_name).replace('\\', '/'))
            points_ranged = pc_range(points,rail_range)
            # points_labels,part_index = label_points(points_ranged,labels,parameter,rail_ranges)
            points_labels,part_index,line_set_para = label_points_lr(points_ranged,labels,parameter_lr,rail_ranges)

            show_all = []
            for part_i in part_index:
                part_point = points_ranged[part_i]
                try:
                    print('part point max:',part_point.shape,np.min(part_point[:,2]),np.max(part_point[:,2]))
                except:
                    pass
                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(np.dot(part_point[:, 0:3],trans_xyz))
                aabb = pcd_part.get_axis_aligned_bounding_box()
                aabb.color = (0, 0, 0)
                show_all.append(aabb)
            new_data = np.concatenate((points_ranged, points_labels[:, np.newaxis]), axis=-1)

            colors = pc_colors(points_labels)
            print(new_data.shape)
            pcd_new = o3d.geometry.PointCloud()
            # pcd_new.points = o3d.utility.Vector3dVector(points_ranged[:, 0:3])
            pcd_new.points = o3d.utility.Vector3dVector(np.dot(points_ranged[:, 0:3],trans_xyz))
            pcd_new.colors = o3d.utility.Vector3dVector(colors.squeeze())
            show_all.append(pcd_new)

            line_set_lines = [[0, 1],[1, 2],[0, 2]]
            quad_colors = [[0, 0.5, 0.5] for i in range(len(line_set_lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.dot(line_set_para,trans_xyz)),
                lines=o3d.utility.Vector2iVector(line_set_lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(quad_colors)
            show_all.append(line_set)

            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh.scale(1, center=mesh.get_center())
            show_all.append(mesh)
            o3d.visualization.draw_geometries(show_all, window_name=file_name + '--' + str(i),
                                              width=1080, height=1080)
            # save label parameter
            np.save(os.path.join(save_npy_root, file_name[:-4]).replace('\\', '/'), new_data)

            ranges_np = np.asarray(rail_ranges)
            labels_np = np.asarray(labels)
            labels_np = labels_np[:, np.newaxis]
            parameter_np = np.asarray(parameter_lr)
            parameter_np = parameter_np[:, np.newaxis]

            parameter_save = np.concatenate((ranges_np, labels_np,parameter_np), axis=-1)
            # parameter_save = np.concatenate((parameter_save, number2_np), axis=-1)
            np.savetxt(os.path.join(save_para_root, file_name[:-4]).replace('\\', '/') + '.txt', parameter_save, fmt='%0.8f')
            # o3d_paint(points_ranged,color=False,name=file_name)

    print(len(files_name))