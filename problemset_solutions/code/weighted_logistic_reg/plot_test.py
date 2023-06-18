import matplotlib.pyplot as plt
import torch
import numpy as np

# Generate a sample binary square matrix
matrix = torch.randint(0, 2, (50, 50))  # Replace with your 50x50 binary matrix

# Create a colormap with blue for 0 and orange for 1
cmap = plt.cm.colors.ListedColormap(['blue', 'orange'])

# Plot the matrix using imshow
plt.imshow(matrix, cmap=cmap, interpolation='nearest')

# Generate random x and y coordinates for markers
x_coords = np.random.randint(0, 50, 20)  # Replace '20' with desired number of markers
y_coords = np.random.randint(0, 50, 20)  # Replace '20' with desired number of markers

# Plot "x" markers
plt.scatter(x_coords, y_coords, marker='x', color='red', s=50)

# Plot "o" markers
plt.scatter(x_coords, y_coords, marker='o', color='green', facecolors='none', s=50)

# Customize plot attributes
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.grid(True, color='black', linewidth=0.5)  # Add grid lines
plt.show()
