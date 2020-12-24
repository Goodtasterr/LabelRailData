import open3d as o3d
import numpy as np
import os
import time
import os
import copy

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

#pcd[N,6]: xyz time intensity flag --> points[N,4]:xyz itensity
def pcd2points():
    root = '/media/hwq/g/qxdpcdascii/a1/'
    files = os.listdir(root)
    files.sort()
    # print(files)

    for file in files:
        f = open(os.path.join(root,file), 'r',encoding='utf8')
        data = f.readlines()

        #number of points
        number_points = int(data[9].split(' ')[-1])
        print('number of points :', number_points)
        if len(data) == number_points+11:
            print('number of points ckecked!')
        else:
            print('data error!')

        #points size [N,4]
        points=np.zeros((number_points,4))
        for i, point_str in enumerate(data[11:]):
            point_data = point_str.split(' ')
            for j,k in enumerate([0,1,2,4]): #x,y,z,i
                points[i][j] = float(point_data[k])


        print((points)[:10])
        exit()

def pcd2xyzi(path):
    f = open(path, 'r', encoding='utf8')
    data = f.readlines()

    # number of points
    number_points = int(data[9].split(' ')[-1])
    print('number of points :', number_points)
    if len(data) == number_points + 11:
        print('number of points ckecked!')
    else:
        print('data error!')

    # points size [N,4]
    points = np.zeros((number_points, 4))
    for i, point_str in enumerate(data[11:]):
        point_data = point_str.split(' ')
        for j, k in enumerate([0, 1, 2, 4]):  # x,y,z,i
            points[i][j] = float(point_data[k])
    return points


#pcd files with motion and static status. motion: 5 frames/s; static: 5 frames/state
def train_files():

    root = '/media/hwq/g/qxdpcdascii/'
    files_a =sorted(os.listdir(root))
    print(files_a)
    # get static or motion state index
    static_ss = [[0, 291, 1058, 1330],
                 [591, 676],
                 [362, 453, 1320, 1562],
                 [0, 855],
                 [0, 77, 403, 466, 1107, 1168, 2018, 2288],
                 [391, 497],
                 [0, 523, 1281, 1483]]
    # static_ss = []
    # for file_a in files_a:
    #     files = sorted(os.listdir(os.path.join(root,file_a)))
    #     if file_a == 'a1':
    #         static_a1_start1 = 0
    #         static_a1_stop1 = files.index('1587955583937334.pcd')
    #         static_a1_start2 = files.index('1587955660085822.pcd')
    #         static_a1_stop2 = len(files) - 1
    #         static_ss.append([0,static_a1_stop1,static_a1_start2,static_a1_stop2])
    #     elif file_a == 'a2':
    #         static_a2_start1 = files.index('1587956121043415.pcd')
    #         static_a2_stop1 = len(files)-1
    #         static_ss.append([static_a2_start1,static_a2_stop1])
    #     elif file_a == 'a3':
    #         static_a3_start1 = files.index('1587957638375780.pcd')
    #         static_a3_stop1 = files.index('1587957647404003.pcd')
    #         static_a3_start2 = files.index('1587957733420029.pcd')
    #         static_a3_stop2 = len(files) - 1
    #         static_ss.append([static_a3_start1, static_a3_stop1,static_a3_start2,static_a3_stop2])
    #     elif file_a == 'a4':
    #         static_a4_start1 = 0
    #         static_a4_stop1 = len(files) - 1
    #         static_ss.append([static_a4_start1, static_a4_stop1])
    #     elif file_a == 'a5':
    #         static_a5_start1 = 0
    #         static_a5_stop1 = files.index('1587957288896700.pcd')
    #         static_a5_start2 = files.index('1587957321249682.pcd')
    #         static_a5_stop2 = files.index('1587957327501607.pcd')
    #         static_a5_start3 = files.index('1587957391110092.pcd')
    #         static_a5_stop3 = files.index('1587957397162750.pcd')
    #         static_a5_start4 = files.index('1587957483988398.pcd')
    #         static_a5_stop4 = len(files) - 1
    #         static_ss.append([static_a5_start1, static_a5_stop1,static_a5_start2,static_a5_stop2,
    #                         static_a5_start3, static_a5_stop3,static_a5_start4,static_a5_stop4])
    #     elif file_a == 'a6':
    #         static_a6_start1 = files.index('1587955852271494.pcd')
    #         static_a6_stop1 = len(files) - 1
    #         static_ss.append([static_a6_start1, static_a6_stop1])
    #     elif file_a == 'a7':
    #         static_a7_start1 = 0
    #         static_a7_stop1 = files.index('1587955310478158.pcd')
    #         static_a7_start2 = files.index('1587955388531322.pcd')
    #         static_a7_stop2 = len(files) - 1
    #         static_ss.append([0, static_a7_stop1,static_a7_start2,static_a7_stop2])

    #get file downsample

    for i,file_a in enumerate(files_a[:-2]):
        #get all files name
        files = sorted(os.listdir(os.path.join(root,file_a)))
        files_name = []
        indexes = []

        static_s = static_ss[i]
        key_static = 0
        for j in range(len(static_s)):
            if j %2==0: #motion state
                if key_static !=static_s[j]:
                    index = np.linspace(key_static,static_s[j],int((static_s[j]-key_static)/2),endpoint=False).astype(np.int)
                    indexes.append(index)
            else: #static state
                index = np.linspace(key_static, static_s[j],5, endpoint=False).astype(np.int)
                indexes.append(index)
            key_static = static_s[j]

        if key_static != len(files):
            index = np.linspace(key_static, len(files), int((len(files) - key_static) / 2), endpoint=False).astype(np.int)
            indexes.append(index)

        # print(np.array(np.concatenate(indexes,axis=0)))
        file_name = ([files[a] for a in (np.concatenate(indexes,axis=0))])
        np.savetxt(os.path.join(root,'train_files',file_a)+'.txt',file_name,fmt = '%s')


if __name__ == '__main__':
    root = '/media/hwq/g/qxdpcdascii/'
    train_index_root = os.path.join(root,'train_index')
    train_file_root = 'a3'  # pcd files in a1
    save_npy_root = os.path.join(root,'labeled_rail','pc_npy',train_file_root)
    save_para_root = os.path.join(root,'labeled_rail','parameter',train_file_root)
    os.makedirs(save_npy_root) if not os.path.exists(save_npy_root) else None
    os.makedirs(save_para_root) if not os.path.exists(save_para_root) else None

    files_name = np.loadtxt(os.path.join(train_index_root,train_file_root+'.txt'),dtype=str)

    rail_range = [-5.5, 7, -30, 30, 6.5, 70] #去掉范围外的离散点

    dist = 0
    speed = 1.8
    #多目标 位置区间　第一行为轨道，之后都是电线杆
    rail_ranges = [[-5.5, -2.4, -0.2, 3.9, 6, 70],  #轨道

              [  1,   6.5, -2.6,   3.1, 70,   70],  # 电线杆上
              [-2.8,  1,    2.1,    3.3,  70, 70],  # 电线杆１

              [-2.9,  1,    1.7,    3.3,  70, 70],  # 电线杆２

              [-2.3,  1,    2.0,    3.3,  70, 70],  # 电线杆3
              [-1.8,6.5, -3, 9, 70,70],  # 电线杆左前
              ]
    labels = [1,2,2,2,2,2] #label parameter 1 : label value

    # [a,_,b,_,_]  直道是一次函数n=1，弯道是二次函数n>1
    #shape of number is same as number2 and rail_range.shape[1]
    parameter = [-0.006,0,-9.6/70,15,22,0] #label parameter 2 : y=a*z^2 + b*z

    for i, file_name in enumerate(files_name):
        if i>=47+dist:
            print('Labeling NO.%d file: %s...in part %s with %d files... '%(i,file_name,train_file_root,len(files_name)))
            points = pcd2xyzi(os.path.join(root,train_file_root,file_name))
            points_ranged = pc_range(points,rail_range)
            points_labels,part_index = label_points(points_ranged,labels,parameter,rail_ranges)

            show_all = []
            for part_i in part_index:
                part_point = points_ranged[part_i]
                try:
                    print('part point max:',part_point.shape,np.min(part_point[:,2]),np.max(part_point[:,2]))
                except:
                    pass
                pcd_part = o3d.geometry.PointCloud()
                pcd_part.points = o3d.utility.Vector3dVector(part_point[:, 0:3])
                aabb = pcd_part.get_axis_aligned_bounding_box()
                aabb.color = (0, 0, 0)
                show_all.append(aabb)
            new_data = np.concatenate((points_ranged, points_labels[:, np.newaxis]), axis=-1)

            colors = pc_colors(points_labels)
            print(new_data.shape)
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(points_ranged[:, 0:3])
            pcd_new.colors = o3d.utility.Vector3dVector(colors.squeeze())
            show_all.append(pcd_new)
            quad_points = [
                [-3.5, -4, parameter[3]],
                [1, -4, parameter[3]],
                [-3.5, 4, parameter[3]],
                [1, 4, parameter[3]],
                [-3.5, -4, parameter[4]],
                [1, -4, parameter[4]],
                [-3.5, 4, parameter[4]],
                [1, 4, parameter[4]],
            ]

            quad_lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],
                        [5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
            quad_colors = [[0, 0.5, 0.5] for i in range(len(quad_lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(quad_points),
                lines=o3d.utility.Vector2iVector(quad_lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(quad_colors)
            show_all.append(line_set)
            o3d.visualization.draw_geometries(show_all, window_name=file_name + '--' + str(i),
                                              width=1080, height=1080)
            # save label parameter
            np.save(os.path.join(save_npy_root, file_name[:-4]), new_data)

            ranges_np = np.asarray(rail_ranges)
            labels_np = np.asarray(labels)
            labels_np = labels_np[:, np.newaxis]
            parameter_np = np.asarray(parameter)
            parameter_np = parameter_np[:, np.newaxis]

            parameter_save = np.concatenate((ranges_np, labels_np,parameter_np), axis=-1)
            # parameter_save = np.concatenate((parameter_save, number2_np), axis=-1)
            np.savetxt(os.path.join(save_para_root, file_name[:-4]) + '.txt', parameter_save, fmt='%0.8f')
            # o3d_paint(points_ranged,color=False,name=file_name)

    print(len(files_name))