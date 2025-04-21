import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Step 1: Calculate coefficients (slope and intercept)
def linear_regression_fit(X, y):
    n = len(X)
    mean_x = np.mean(X)
    mean_y = np.mean(y)

    numerator = np.sum((X - mean_x) * (y - mean_y))
    denominator = np.sum((X - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept

slope, intercept = linear_regression_fit(X, y)
print(f" Slope: {slope:.2f}, Intercept: {intercept:.2f}")

# Step 2: Predict
def predict(X, slope, intercept):
    return slope * X + intercept

y_pred = predict(X, slope, intercept)

# Step 3: Evaluate
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

print(f" MSE: {mean_squared_error(y, y_pred):.4f}")
print(f" R² Score: {r2_score(y, y_pred):.4f}")

# Step 4: Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
