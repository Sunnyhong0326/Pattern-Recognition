import numpy as np
import matplotlib.pyplot as plt

def generate_data(mean, covariance, N=1000):
    X1 = np.random.multivariate_normal(mean, covariance, N)
    return X1

def generate_bernoulli_and_estimate(p, N):
    X = np.random.binomial(n=1, p=p, size=N)
    p_hat_ML = np.mean(X)
    return X, p_hat_ML

def generate_poly_data(N, train_val_split_ratio=0.8):
    x = np.random.uniform(0, 10, N)
    epsilon = np.random.normal(0, 1, N)
    y = 3 * np.sin(0.8 * x + 2) + epsilon

    train_samples = int(N*0.8)
    train_x = x[:train_samples].reshape(-1, 1)  # 80 samples for training
    train_y = y[:train_samples]
    val_x = x[train_samples:].reshape(-1, 1)    # 20 samples for validation
    val_y = y[train_samples:]

    return train_x, train_y, val_x, val_y

# Function to expand x to polynomial features
def polynomial_features(x, degree):
    return np.hstack([x**i for i in range(degree+1)])

# Function to fit polynomial using Normal Equation
def fit_polynomial(x, y, degree):
    X_poly = polynomial_features(x, degree)
    # Normal Equation: w = (X^T X)^(-1) X^T y
    w = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return w

def calculate_mse(w, x, y, degree):
    X_poly = polynomial_features(x, degree)
    y_pred = X_poly @ w
    mse = np.mean((y - y_pred)**2)
    return mse

def plot_regression_result(train_x, train_y, degrees=[1, 3, 5, 7, 9]):
    plt.figure(figsize=(10, 6))
    plt.scatter(train_x, train_y, color='blue', label='Training Data', zorder=5)

    for k in degrees:
        # Fit the polynomial to the training data
        w = fit_polynomial(train_x, train_y, k)
        
        # Generate x values for plotting the fitted curve
        x_fit = np.linspace(0, 10, 100).reshape(-1, 1)
        y_fit = np.sum([w[i] * x_fit**i for i in range(k+1)], axis=0)
        
        # Plot the fitted curve
        plt.plot(x_fit, y_fit, label=f"Degree {k}")

    plt.title("Training Data and Fitted Polynomial Curves")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot_error(train_x, train_y, val_x, val_y, degrees=[1, 3, 5, 7, 9]):
    val_errors = []
    train_errors = []

    for k in degrees:
        # Fit the polynomial to the training data
        w = fit_polynomial(train_x, train_y, k)
        
        train_mse = calculate_mse(w, train_x, train_y, k)
        # Calculate MSE on validation set
        val_mse = calculate_mse(w, val_x, val_y, k)
        val_errors.append(val_mse)
        train_errors.append(train_mse)

    # Plot Error vs Polynomial Order
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, val_errors, marker='o', color='red', linestyle='-', label="Validation Error (MSE)")
    plt.plot(degrees, train_errors, marker='o', color='green', linestyle='-', label="Train Error (MSE)")
    plt.title("Validation Error vs Polynomial Order")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.show()