# 3D_Data_Starter
# Basic plotting of data in 3D

# Import the matplotlib package and alias the pyplot module 
import matplotlib.pyplot as plt
import numpy as np

# Library for 3D plots
from mpl_toolkits.mplot3d import Axes3D

# Create the plot object
fig = plt.figure()
ax = plt.axes(projection='3d')

xdata  =[]
ydata = []
zdata = []

# Read in the data
# Data format: x,y,z

f = open("3D_data.txt")
data = f.readlines()
f.close()

# Parse the data into lists of x's, y's, and z's
for i in range(len(data)):
  datapoint = data[i].split(',')
  xdata.append(float(datapoint[0]))
  ydata.append(float(datapoint[1]))
  zdata.append(float(datapoint[2]))

# Create the 3D plot
ax.scatter3D(xdata,ydata,zdata, c=zdata, cmap = 'Greens')

# Display the plot object
# Commented out because it does not work in GitHub Codespaces
#plt.show()

# Save the plot object as a png
plt.savefig('3D_Data_output.png')

