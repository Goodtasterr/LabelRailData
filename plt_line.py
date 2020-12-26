import numpy as np
import matplotlib.pyplot as plt
import math
import copy
def rail_fit(parameters):

        k1,b1,k2,b2,r,delta_r = parameters
        x0y0 = np.asarray([(b1-b2)/(k2-k1),(b1*k2-k1*b2)/(k2-k1)]).astype(np.float32)
        theta = np.zeros(10).astype(np.float32)
        theta[1] = math.atan(k1)
        theta[2] = math.atan(k2)
        theta[3] = theta[2] -theta[1]
        theta[4] = 0.5*(math.pi-(np.sign(k2).astype(np.float32))*theta[3])

        delta_L = r/math.tan(theta[4])
        x1 = x0y0[0]-delta_L*math.cos(theta[1])
        x2 = x0y0[0]+delta_L*math.cos(theta[2])

        x_range = np.arange(x1-1,x2+1,0.1).astype(np.float32)
        y1 = k1*x_range+b1
        y2 = k2*x_range+b2


        xryr = [x0y0[0]-r*math.cos(theta[4]-(np.sign(k2))*theta[1])/math.sin(theta[4]),
                x0y0[1]+(np.sign(k2))*r*math.sin(theta[4]-(np.sign(k2))*theta[1])/math.sin(theta[4])]
        xryr = np.asarray(xryr).astype(np.float32)

        #xy coordinate
        x3 = np.arange(x1,x2,0.01)
        y3 = xryr[1]-(np.sign(k2))*(np.sqrt(r**2-(x3-xryr[0])**2))

        ### left rail
        y1_1 = k1*x_range+b1+delta_r*(1+k1**2)**0.5
        y2_1 = k2*x_range+b2+delta_r*(1+k2**2)**0.5
        x1_1 = x1-delta_r*math.sin(theta[1])
        x2_1 = x2-delta_r*math.sin(theta[2])

        #xy coordinate
        x3_1 = np.arange(x1_1,x2_1,0.01)
        y3_1 = xryr[1]-(np.sign(k2))*(np.sqrt((r-(np.sign(k2))*delta_r)**2-(x3_1-xryr[0])**2))

        line_set_para = np.zeros([3,3])
        line_set_para[0] = [0,xryr[1],xryr[0]]
        line_set_para[1] = [0,k1*x1+b1,x1]
        line_set_para[2] = [0, k2 * x2 + b2, x2]

        return np.asarray([xryr[0],xryr[1],x1,x2,x1_1,x2_1]).astype(np.float32),line_set_para


def label_points_lr(points,labels,parameter,ranges):
    label = np.zeros([points.shape[0]]).astype(int)
    part_index = []
    k1, b1, k2, b2, r, delta_r = parameter
    (xr, yr, x1, x2, x1_1, x2_1) ,line_set_para= rail_fit(parameter) #xr yr x1 x2 x1_1 x1_2
    print(parameter,xr, yr, x1, x2, x1_1, x2_1)
    for i,range in enumerate(ranges):
        ranged = copy.deepcopy(range)
        if i ==0:
            idx1 = (points[:,2]<=x1)
            idx2 = (points[:,2]<=x2) &(points[:,2]>x1)
            idx3 = (points[:,2]>x2)
            idx1_1 = (points[:,2]<=x1_1)
            idx2_1 = (points[:,2]<=x2_1) &(points[:,2]>x1_1)
            idx3_1 = (points[:,2]>x2_1)
            ranged[2] = idx1 * (k1*points[:,2]+b1)
            # yr-0.1 是调试加入0.1
            ranged[2] += idx2 * (yr-0.1-(np.sign(k2))*(np.sqrt(r**2-((points[:,2]-xr)*idx2)**2)))
            ranged[2] += idx3 * (k2*points[:, 2] + b2)

            ranged[3] = idx1_1*(k1*points[:,2]+b1+delta_r*(1+k1**2)**0.5)
            ranged[3] += idx2_1*(yr-(np.sign(k2))*(np.sqrt((r-(np.sign(k2))*delta_r)**2-((points[:,2]-xr)*idx2_1)**2)))
            ranged[3] += idx3_1 * (k2 * points[:, 2] + b2 + delta_r * (1 + k2 ** 2) ** 0.5)

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
    return label, part_index, line_set_para

if __name__ == '__main__':

    #k1,b1,k2,b2,r
    parameters = np.asarray([-0.006,-2,-0.135,5,22,4]).astype(np.float32)
    # parameters = np.asarray([-0.001,-1,0.5,2,1,0.4]).astype(np.float32)
    k1,b1,k2,b2,r,delta_r = parameters
    x0y0 = np.asarray([(b1-b2)/(k2-k1),(b1*k2-k1*b2)/(k2-k1)]).astype(np.float32)
    theta = np.zeros(10).astype(np.float32)
    theta[1] = math.atan(k1)
    theta[2] = math.atan(k2)
    theta[3] = theta[2] -theta[1]
    theta[4] = 0.5*(math.pi-(np.sign(k2).astype(np.float32))*theta[3])

    delta_L = r/math.tan(theta[4])
    x1 = x0y0[0]-delta_L*math.cos(theta[1])
    x2 = x0y0[0]+delta_L*math.cos(theta[2])

    x_range = np.arange(x1-1,x2+1,0.1).astype(np.float32)
    y1 = k1*x_range+b1
    y2 = k2*x_range+b2


    xryr = [x0y0[0]-r*math.cos(theta[4]-(np.sign(k2))*theta[1])/math.sin(theta[4]),
            x0y0[1]+(np.sign(k2))*r*math.sin(theta[4]-(np.sign(k2))*theta[1])/math.sin(theta[4])]


    xryr = np.asarray(xryr).astype(np.float32)

    #xy coordinate
    x3 = np.arange(x1,x2,0.01)
    y3 = xryr[1]-(np.sign(k2))*(np.sqrt(r**2-(x3-xryr[0])**2))

    # # r theta coordinate
    # if k2>=0:
    #     theta_r = np.arange(1.5*math.pi+theta[1],1.5*math.pi+theta[2],0.01) #trun left k2>0
    # else:
    #
    #     theta_r = np.arange(0.5*math.pi+theta[2],0.5*math.pi+theta[1],0.01)  #turn right k2<0
    #
    # x3 = xryr[0]+r*np.cos(theta_r)
    # y3 = xryr[1]+r*np.sin(theta_r)

    ### left rail
    y1_1 = k1*x_range+b1+delta_r*(1+k1**2)**0.5
    y2_1 = k2*x_range+b2+delta_r*(1+k2**2)**0.5
    x1_1 = x1-delta_r*math.sin(theta[1])
    x2_1 = x2-delta_r*math.sin(theta[2])

    #xy coordinate
    x3_1 = np.arange(x1_1,x2_1,0.01)
    y3_1 = xryr[1]-(np.sign(k2))*(np.sqrt((r-(np.sign(k2))*delta_r)**2-(x3_1-xryr[0])**2))

    # x3_1 = xryr[0]+(r-(np.sign(k2))*delta_r)*np.cos(theta_r)
    # y3_1 = xryr[1]+(r-(np.sign(k2))*delta_r)*np.sin(theta_r)
    print('return paras:',np.asarray([xryr[0],xryr[1],x1,x2,x1_1,x2_1]).astype(np.float32)  )

    plt.title("rail lines")
    plt.xlabel("x axis --> pointcloud[2]")
    plt.ylabel("y axis --> pointcloud[1]")
    fig = plt.figure()
    axes = fig.add_subplot()
    axes.axis('equal')

    axes.plot(x_range,y1)
    axes.plot(x_range,y2)
    axes.plot(x3,y3)

    axes.plot(x_range,y1_1)
    axes.plot(x_range,y2_1)
    axes.plot(x3_1,y3_1)


    print(xryr[0]+r*np.cos(2*math.pi-theta[4]+theta[1]),xryr[1]+r*np.sin(2*math.pi-theta[4]+theta[1]))
    keypoints = []
    keypoints.append(xryr)
    # keypoints.append(x0y0)
    keypoints.append((x1,k1*x1+b1))
    keypoints.append((x2,k2*x2+b2))
    keypoints.append((x1_1,k1*x1_1+b1+delta_r*(1+k1**2)**0.5))
    keypoints.append((x2_1,k2*x2_1+b2+delta_r*(1+k2**2)**0.5))

    keypoints = np.asarray(keypoints).astype(np.float32)
    axes.scatter(keypoints[:,0],keypoints[:,1],marker='x',color='red',s=40)

    plt.show()