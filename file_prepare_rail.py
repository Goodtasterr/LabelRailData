import open3d as o3d
import numpy as np
import time
import os
import copy




#pcd[N,6]: xyz time intensity flag --> points[N,4]:xyz itensity
def pcd2points():
    root = '/media/hwq/g/qxdpcdascii/a1/'
    files = os.listdir(root)
    files.sort()
    # print(files)

    for file in files:
        f = open(os.path.join(root,file).replace('\\', '/'), 'r',encoding='utf8')
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
    #     files = sorted(os.listdir(os.path.join(root,file_a).replace('\\', '/')))
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
        files = sorted(os.listdir(os.path.join(root,file_a).replace('\\', '/')))
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
        np.savetxt(os.path.join(root,'train_files',file_a).replace('\\', '/')+'.txt',file_name,fmt = '%s')
