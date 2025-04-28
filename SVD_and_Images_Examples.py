# Image Diagonalization
# Patrick Honner, 4/23/2022

# Image library
from PIL import Image, ImageOps

# Scientific computing library
import numpy as np

# Plotting library
import matplotlib.pyplot as plt


# Instantiate the plot object with two subplots
fig, axs = plt.subplots(2)

# Load image in from file
im = Image.open("./PIH.jpg")
width, height = im.size

# Resize the image for testing purposes
#w = int(width/50)
#h = int(height/50)
#n = 30
#im2 = im.resize((w,h))

im2 = im

# Converts the image to grayscale
im2 = ImageOps.grayscale(im2)

# Load the image data into an array
data = np.asarray(im2)

# Get the SVD information from the matrix
# u and vh are the lists of vectors
# s is the *sorted* list of singular values
u, s, vh = np.linalg.svd(data, full_matrices=True,compute_uv=True)

# Initialize the Approximation array
n = len(s)
matrix_approx_2 = np.zeros((height,width)) 


# Create a rank K approximation
K = 10

for i in range(K):
  matrix_approx_2 += s[i]*np.outer(u[:,i],vh[i])

rows, columns = matrix_approx_2.shape

# Some entries end up summing to above the maximum grayscale value ?
# This caps each entry
# (Trying commenting this out and see what happens)
for i in range(rows):
  for j in range(columns):
    if matrix_approx_2[i][j] > 255:
      matrix_approx_2[i][j]=255

# This converts the entires of the data matrix into a form recognizable by the image object 
Approx = matrix_approx_2.astype(np.uint8)

# Creates a new image from the data array
im3 = Image.fromarray(Approx,'L')

# Set the zerorth subplot to the original image
axs[0].imshow(im2,cmap = plt.cm.gray)

# Set first subplot to the manipulated image
axs[1].imshow(im3,cmap = plt.cm.gray)

plt.savefig('SVD_Example.png')


