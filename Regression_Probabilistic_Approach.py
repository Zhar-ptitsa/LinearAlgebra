# Regression Lines -- Probabilistic Approach
# Patrick Honner, 3/19/23

# Math libraries
import numpy as np
import math

# Import the matplotlib package and alias the pyplot module 
import matplotlib.pyplot as plt

# Create the plot object
fig, ax = plt.subplots()

# Set the viewing window in the plot from -50 to 50  in both directions
ax.set_xlim(-50,50)
ax.set_ylim(-50,50)

# Function for computing the norm of a row vector
def norm(v):
  norm_v = 0
  for a in v[0]:
    norm_v += a*a
  return(math.sqrt(norm_v))


# Read in the data
# Data format: x,y

xdata = []
ydata = []

f = open("2D_data.txt")
data = f.readlines()
f.close()

# Parse the data into lists of x's and y's

for i in range(len(data)):
  datapoint = data[i].split(',')
  xdata.append(float(datapoint[0]))
  ydata.append(float(datapoint[1]))

# Create an all 1s vector 
ones_vec = np.ones(len(xdata))

# Create the matrices for Ax = b

# Create the matrix A transpose (it's easier to work with row vectors)
A_T = np.array([xdata,ones_vec])
print(A_T)
print("\n")

# Create the b tranpose vector
y_vals_T = np.array([ydata])
print(y_vals_T)
print("\n\n")


# An initial guess at the coefficients y = Ax + B
coeffs_T = np.array([0.1,0.1])
error = 1000000
initial_step_size = 2.0
final_step_size = 0.00001

step_size = initial_step_size

# Basic Algorithm:
# 1) Compute the error of y = Ax + B associated witht the guess A, B
# 2) Take a random jump in AB-space and compute the new error
# 3) If error is reduced, keep the new A and B. If error is not reduced, go back and take another jump
# 4) After 10 failed jumps (that do not reduce error), cut the step size in half
# 5) End when step_size hits final_step_size

while (step_size > final_step_size):
  step_size = step_size / 2.0
  # Count the number of "failed" guesses; i.e., guesses that don't reduce error
  num_guesses = 0

  while (num_guesses < 10):
 
    # Take a random jump in AB space and compute the new error
    theta = np.random.uniform(0,2*math.pi)
    jump = np.array([step_size*math.cos(theta),step_size*math.sin(theta)])
    test_coeffs_T = np.copy(coeffs_T + jump)
    D  = test_coeffs_T@A_T - y_vals_T
    test_error = norm(D)*norm(D)

    # If error is reduced, keep the new A,B pair and re-start the jump-sequence
    if (test_error < error):

      # Add line to plot to show progression
      # alpha is opacity of line
      plt.plot([-50,50], [-50*coeffs_T[0]+coeffs_T[1], 50*coeffs_T[0]+coeffs_T[1]],color ='b', alpha = max(0.1,0.5-step_size))
      error = test_error
      num_guesses = 0
      coeffs_T = np.copy(test_coeffs_T)
    else:
      num_guesses += 1 


print("Regression line:   y=",coeffs_T[0],"x + ", coeffs_T[1])
  
# This plots the points (xdata[i], ydata[i])
# The 'ro' option yields red unconnected dots

plt.plot(xdata,ydata,'ro')

# The following code plots a line

plt.plot([-50,50], [-50*coeffs_T[0]+coeffs_T[1], 50*coeffs_T[0]+coeffs_T[1]],'b', ls = '--')

# This shows the plot object (Must ctrl-c out of this in the console)
# Commented out because it does not work in GitHub Codespaces
# plt.show() 

# Save plot to png
plt.savefig('probabilistic_regression_output.png')

# It's pretty close! https://www.desmos.com/calculator/r6bhgm3g8r

