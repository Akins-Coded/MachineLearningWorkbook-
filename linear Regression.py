from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()  # Instantiate the model
model.fit(X_train, Y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, Y, alpha=0.7, label='Original Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Linear Regression In Python')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")
