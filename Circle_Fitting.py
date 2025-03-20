# Circle Fitting
# Patrick Honner, 3/19/23
# Uses an error-reducing, iterative algorithm to fit a circle to a given set of points

import numpy as np
import matplotlib.pyplot as plt

# Create the plot object
fig, ax = plt.subplots()

# This variable tracks the number of "operations" performed in finding the intersection
# Here an operation is defined to be an addition, multiplication, comparison, or assignment
operations = 0

# The hardcoded points to fit a circle to
x_coords = [1,3, 7, 10]
y_coords = [4,-5,5,1]

# The plot bounds
x_win_lower = -15
x_win_upper = 25
y_win_lower = -20
y_win_upper = 20

def f(x,y,A,B,C):
    # f(x,y) = x^2 + y^2 + Ax + By + C
    global operations
    operations += 8
    return (x**2 + y**2 + A*x + B*y + C)

def compute_error(A,B,C):
    global operations
    operations += len(x_coords)
    error = 0
    for i in range(len(x_coords)):
        error += (f(x_coords[i], y_coords[i],A,B,C))**2
    return error


# Plot each circle
def plot_circle(A,B,C):
    x_range = np.linspace(x_win_lower, x_win_upper, 1000)
    y_range_1 = B/2 + np.sqrt( (A**2)/4 + (B**2)/4 - C - (x_range + (A/2))**2)
    y_range_2 = B/2 - np.sqrt( (A**2)/4 + (B**2)/4 - C - (x_range + (A/2))**2)
    x_range = np.concatenate((x_range, x_range))
    y_range = np.concatenate((y_range_1, y_range_2))
    return x_range, y_range


#init_guess = [20*np.random.random(1) ,20*np.random.random(1), 20*np.random.random(1)]
init_guess = [-17, 8, -10]
guess = init_guess

A_guesses, B_guesses, C_guesses = [], [], []

current_error = 1000000
new_error = current_error + 1

tolerance = 300
step_multiplier = 1


while current_error > tolerance:
    operations += 1
    num_direction_guesses = 0

    while current_error <= new_error:

        operations += 1

        num_direction_guesses +=1
        operations +=1

        # Take a step in a random direction, compute the associated error, repeat until error is reduced
        alpha = np.random.uniform(0,2*np.pi)
        beta = np.random.uniform(0,2*np.pi)
        operations += 2
        new_guess = [guess[0] + step_multiplier*np.cos(alpha)*np.sin(beta), guess[1] + step_multiplier*np.cos(alpha)*np.cos(beta), guess[2] + step_multiplier*np.sin(alpha)]
        new_error = compute_error(new_guess[0],new_guess[1], new_guess[2])
        
        if (num_direction_guesses > 7):
            step_multiplier = 0.75 * step_multiplier
            num_direction_guesses= 0
            operations += 3

    print(guess[0], guess[1], guess[2], current_error)

                   
    guess = new_guess
    A_guesses.append(guess[0])
    B_guesses.append(guess[1])
    C_guesses.append(guess[2])

    current_error = new_error

    operations += 4



plt.plot(x_coords,y_coords, 'ro')
ax.set_xlim(x_win_lower, x_win_upper)
ax.set_ylim(y_win_lower, y_win_upper)
plt.gca().set_aspect('equal', adjustable='box')
ax.set_yticklabels([])
ax.set_xticklabels([])

for i in range(len(A_guesses)):
    xdata, ydata = [], []
    xdata, ydata = plot_circle(A_guesses[i],B_guesses[i],C_guesses[i])
    color_string = float(1 - (i/len(A_guesses)))
    plt.plot(xdata, ydata, color = str(color_string) )

# Show the plot object
# Commented out because it does not work in GitHub Codespaces
#plt.show()

# Save plot as png
plt.savefig('circle_fitting_output.png')
