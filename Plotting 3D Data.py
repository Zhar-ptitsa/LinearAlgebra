# Import the matplotlib package and alias the pyplot module 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create the plot object
fig = plt.figure()
ax = plt.axes(projection='3d')

# Create the data
# These are the principal axes
principal_1 = [1,2,2]
principal_2 = [-2,2,-1]
principal_3 = [-2,-1,2]

# These are the weights along each principal axis
p1_factor = 30
p2_factor = 20
p3_factor = 5

xdata=[]
ydata=[]
zdata=[]

for i in range(50):
  a = np.random.random()-0.5
  b = np.random.random() - 0.5
  c = np.random.random() - 0.5

  xdata.append(a*p1_factor*principal_1[0]+b*p2_factor*principal_2[0] + c*p3_factor*principal_3[0])
  ydata.append(a*p1_factor*principal_1[1]+b*p2_factor*principal_2[1] + c*p3_factor*principal_3[1])
  zdata.append(a*p1_factor*principal_1[2]+b*p2_factor*principal_2[2] + c*p3_factor*principal_3[2])

f = open("3D_data.txt","w")

for i in range(50):
  f.write('{},{},{}\n'.format(xdata[i],ydata[i],zdata[i]))
f.close()


ax.scatter3D(xdata,ydata,zdata, c=zdata, cmap = 'Greens')

plt.show()

