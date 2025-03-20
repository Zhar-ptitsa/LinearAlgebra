# Import the matplotlib package and alias the pyplot module 
import matplotlib.pyplot as plt
import numpy as np

# Create the plot object
fig, ax = plt.subplots()

# Set the viewing window from 0 to 20 in both directions
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)

principal_1 = [1,2]
principal_2 = [-2,1]

p1_factor = 30
p2_factor = 10

xdata=[]
ydata=[]

for i in range(50):
  a = np.random.random()-0.5
  b = np.random.random() - 0.5
  
  xdata.append(a*p1_factor*principal_1[0]+b*p2_factor*principal_2[0])
  ydata.append(a*p1_factor*principal_1[1]+b*p2_factor*principal_2[1])


f = open("2D_data.txt","w")

for i in range(50):
  f.write('{},{}\n'.format(xdata[i],ydata[i]))
f.close()
  

# Here we define the set of x- and y-coordinates beforehand
# The 'ro' option yields red unconnected dots

plt.plot(xdata,ydata, 'r+')

# This plots the line from (0,5) to (20,1), i.e. the line y = -0.2x + 5 
plt.plot([-25,25], [-50,50],'b')

# This shows the object (Must ctrl-c out of this in the console)
plt.show() 
