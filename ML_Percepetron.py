import numpy as np
import matplotlib.pyplot as plt

# Training data: [feature1, feature2]
# Let's say we want to classify points into 2 groups
# The decision boundary will be a curve (not a straight line)
X = np.array([
     [10, 9],
    [1, 2],
    [11, 6],
    [4, 3],
    [8, 7],
    [1, 10],
    [2, 4],
    [9, 8],
    [3, 4],
    [12, 11],
    [5, 4],
    [7, 9],
    [2, 1],
    [10, 5],
    [3, 2],
    [6, 7],
    [8, 9],
    [4, 6],
    [9, 10],
    [11, 12],
    [1, 1],
    [2, 3],
    [6, 5],
    [3, 5],
    [7, 6],
    [10, 8],
    [5, 7],
    [12, 10],
    [9, 11]
])

# 1 is passed, 0 is fail
y = np.array([
    1,0,1,0,1,0,0,1,0,
    1,0,1,0,0,0,1,1,1,1,1,
    0,0,0,1,0,1,1,1,1
])  # labels

# Add polynomial features to create non-linear decision boundary
def add_polynomial_features(X):
    """Add x1^2, x2^2, and x1*x2 features for curved boundary"""
    x1_squared = X[:, 0]**2
    x2_squared = X[:, 1]**2
    x1_x2 = X[:, 0] * X[:, 1]
    return np.column_stack([X, x1_squared, x2_squared, x1_x2])

# Transform training data with polynomial features
X_poly = add_polynomial_features(X)

# Initialize weights and bias (now 5 features instead of 2)
w = np.zeros(X_poly.shape[1])
b = 0
learning_rate = 0.01
epochs = 100

# Activation function: sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training loop
for epoch in range(epochs):
    for i in range(len(X_poly)):
        z = np.dot(X_poly[i], w) + b
        y_pred = sigmoid(z)
        error = y[i] - y_pred
        w += learning_rate * error * X_poly[i]
        b += learning_rate * error
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: weights={w}, bias={b:.4f}")

# Test the model
test_points = np.array([
    [10, 10],
    [2, 1],
    [1, 4],
    [7, 9],
    [5, 5],
    [4, 2],
    [0, 0]
])

print("\nTest results:")
for point in test_points:
    point_poly = add_polynomial_features(point.reshape(1, -1))
    z = np.dot(point_poly, w) + b
    prediction = sigmoid(z).item()
    print(f"Point {point} → sigmoid={prediction:.4f}, class {1 if prediction >= 0.5 else 0}")

# Visualize the curved decision boundary
print("\nGenerating visualization...")
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))

# Create grid of points and add polynomial features
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
grid_poly = add_polynomial_features(grid_points)

# Predict for all grid points
Z = sigmoid(np.dot(grid_poly, w) + b)
Z = Z.reshape(xx1.shape)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot decision boundary as contour (curved)
plt.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)

# Plot training points
passed = X[y == 1]
failed = X[y == 0]
plt.scatter(passed[:, 0], passed[:, 1], c='blue', marker='o', s=100, edgecolors='k', label='Passed')
plt.scatter(failed[:, 0], failed[:, 1], c='red', marker='x', s=100, linewidths=2, label='Failed')

# Plot test points
for point in test_points:
    point_poly = add_polynomial_features(point.reshape(1, -1))
    z = np.dot(point_poly, w) + b
    prediction = sigmoid(z).item()
    color = 'blue' if prediction >= 0.5 else 'red'
    plt.scatter(point[0], point[1], c=color, marker='s', s=150, edgecolors='green', linewidths=3, label='Test' if point is test_points[0] else '')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Perceptron with Curved Decision Boundary\n(Using Polynomial Features)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()