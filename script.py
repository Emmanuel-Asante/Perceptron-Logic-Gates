import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Create a list of points
data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Labels for an OR gate
labels = [0, 1, 1, 1]

# Create a scatter plot
plt.scatter(
  [point[0] for point in data],
  [point[1] for point in data],
  c = labels
)

# Create a Perceptron object (model)
classifier = Perceptron(max_iter = 40)

# Train the model
classifier.fit(data, labels)

# Print out the accuracy of the model
print(classifier.score(data, labels))

# Distances of points from the decision boundary
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

# A list decimal numbers between 0 and 1
x_values = np.linspace(0, 1, 100)

# A list decimal numbers between 0 and 1
y_values = np.linspace(0, 1, 100)

# Create a list of points
point_grid = list(product(x_values, y_values))

# Find distances of points from the decision boundary
distances = classifier.decision_function(point_grid)

# Make all distances positives
abs_distances = [abs(distance) for distance in distances]

# Reshape abs_distances to a 100 X 100 matrix
distances_matrix = np.reshape(abs_distances, (100, 100))

# Create a heat map
heatmap = plt.pcolormesh(
  x_values,
  y_values,
  distances_matrix
)

# Put legend on heatmap
plt.colorbar(heatmap)

# Show plot
plt.show()