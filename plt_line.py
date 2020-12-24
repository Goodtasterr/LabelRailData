import numpy as np
import matplotlib.pyplot as plt
import math
#k1,b1,k2,b2,r
# parameters = np.asarray([0,1,1,-2,2,0.5]).astype(np.float32)
parameters = np.asarray([-0.001,-1,-0.1,2,6,0.4]).astype(np.float32)
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

# xy coordinate
# xr_range = np.arange(x1-0.3,x2+0.3,0.01)
# y3_2 = xryr[1]-(np.sqrt(r**2-(xr_range-xryr[0])**2))

# r theta coordinate
if k2>=0:
    theta_r = np.arange(1.5*math.pi+theta[1],1.5*math.pi+theta[2],0.01) #trun left k2>0
else:

    theta_r = np.arange(0.5*math.pi+theta[2],0.5*math.pi+theta[1],0.01)  #turn right k2<0

x3 = xryr[0]+r*np.cos(theta_r)
y3 = xryr[1]+r*np.sin(theta_r)

### left rail
y1_1 = k1*x_range+b1+delta_r*(1+k1**2)**0.5
y2_1 = k2*x_range+b2+delta_r*(1+k2**2)**0.5
x1_1 = x1-delta_r*math.sin(theta[1])
x2_1 = x2-delta_r*math.sin(theta[2])
x3_1 = xryr[0]+(r-(np.sign(k2))*delta_r)*np.cos(theta_r)
y3_1 = xryr[1]+(r-(np.sign(k2))*delta_r)*np.sin(theta_r)


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
keypoints.append(x0y0)
keypoints.append((x1,k1*x1+b1))
keypoints.append((x2,k2*x2+b2))
keypoints.append((x1_1,k1*x1_1+b1+delta_r*(1+k1**2)**0.5))
keypoints.append((x2_1,k2*x2_1+b2+delta_r*(1+k2**2)**0.5))

keypoints = np.asarray(keypoints).astype(np.float32)
axes.scatter(keypoints[:,0],keypoints[:,1],marker='x',color='red',s=40)

plt.show()