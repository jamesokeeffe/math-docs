"""
Linear Regression Implementation using Normal Equation
Step 1: Linear Algebra - Project 1

Mathematical Foundation:
For linear regression y = Xβ + ε, the optimal parameters are:
β = (X^T X)^(-1) X^T y

This implementation demonstrates:
1. How linear algebra solves optimization problems analytically
2. The role of matrix operations in machine learning
3. Geometric interpretation of least squares
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

class LinearRegressionNormalEquation:
    """
    Linear Regression using the Normal Equation (closed-form solution).
    
    The normal equation provides an analytical solution to linear regression
    without requiring iterative optimization.
    """
    
    def __init__(self):
        self.weights = None
        self.intercept = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionNormalEquation':
        """
        Fit linear regression using normal equation: β = (X^T X)^(-1) X^T y
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        
        Returns:
            self: Fitted estimator
        """
        # Add bias term (intercept) by prepending column of ones
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: β = (X^T X)^(-1) X^T y
        try:
            # Compute X^T X
            XtX = X_with_bias.T @ X_with_bias
            
            # Check for singularity (matrix not invertible)
            if np.linalg.det(XtX) == 0:
                warnings.warn("X^T X is singular. Using pseudo-inverse.")
                XtX_inv = np.linalg.pinv(XtX)
            else:
                XtX_inv = np.linalg.inv(XtX)
            
            # Compute full solution
            beta = XtX_inv @ X_with_bias.T @ y
            
            # Extract intercept and weights
            self.intercept = beta[0]
            self.weights = beta[1:]
            self.is_fitted = True
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra error during fitting: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return X @ self.weights + self.intercept
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        Args:
            X: Feature matrix
            y: True target values
        
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def generate_sample_data(n_samples: int = 100, noise_std: float = 0.1, 
                        random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for linear regression demonstration.
    
    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of noise
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix, y: Target vector
    """
    if random_state:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, 2)
    
    # True relationship: y = 3*x1 + 2*x2 + 1 + noise
    true_weights = np.array([3.0, 2.0])
    true_intercept = 1.0
    
    y = X @ true_weights + true_intercept + np.random.normal(0, noise_std, n_samples)
    
    return X, y


def demonstrate_normal_equation():
    """
    Demonstrate the normal equation solution with visualizations.
    """
    print("=== Linear Regression with Normal Equation ===\n")
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise_std=0.5)
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Fit model
    model = LinearRegressionNormalEquation()
    model.fit(X, y)
    
    print(f"Fitted weights: {model.weights}")
    print(f"Fitted intercept: {model.intercept:.3f}")
    print(f"True weights: [3.0, 2.0]")
    print(f"True intercept: 1.0")
    
    # Evaluate model
    r2_score = model.score(X, y)
    print(f"R² score: {r2_score:.3f}")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Visualize results (for 1D case, we'll use first feature)
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted (R² = {r2_score:.3f})')
    plt.grid(True)
    
    # Plot 2: Feature 1 vs Target
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], y, alpha=0.6, label='Data')
    # For visualization, show the effect of first feature with second feature at mean
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_mean = np.mean(X[:, 1])
    X_viz = np.column_stack([x1_range, np.full_like(x1_range, x2_mean)])
    y_viz = model.predict(X_viz)
    plt.plot(x1_range, y_viz, 'r-', lw=2, label='Fitted line')
    plt.xlabel('Feature 1')
    plt.ylabel('Target')
    plt.title('Feature 1 vs Target')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def matrix_analysis_demo():
    """
    Demonstrate the matrix operations in the normal equation.
    """
    print("\n=== Matrix Analysis of Normal Equation ===\n")
    
    # Simple 2D example for clear matrix demonstration
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([2, 4, 6, 8])
    
    print("Feature matrix X:")
    print(X)
    print("\nTarget vector y:")
    print(y)
    
    # Add bias column
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    print("\nX with bias column:")
    print(X_with_bias)
    
    # Step-by-step normal equation
    print("\n--- Normal Equation Steps ---")
    
    # Step 1: X^T
    Xt = X_with_bias.T
    print("\n1. X^T:")
    print(Xt)
    
    # Step 2: X^T X
    XtX = Xt @ X_with_bias
    print("\n2. X^T X:")
    print(XtX)
    
    # Step 3: (X^T X)^(-1)
    XtX_inv = np.linalg.inv(XtX)
    print("\n3. (X^T X)^(-1):")
    print(XtX_inv)
    
    # Step 4: X^T y
    Xty = Xt @ y
    print("\n4. X^T y:")
    print(Xty)
    
    # Step 5: β = (X^T X)^(-1) X^T y
    beta = XtX_inv @ Xty
    print("\n5. β = (X^T X)^(-1) X^T y:")
    print(beta)
    print(f"   Intercept: {beta[0]:.3f}")
    print(f"   Weights: {beta[1:]}")
    
    # Verify prediction
    y_pred = X_with_bias @ beta
    print(f"\nPredictions: {y_pred}")
    print(f"Actual:      {y}")
    print(f"MSE:         {np.mean((y - y_pred)**2):.6f}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_normal_equation()
    matrix_analysis_demo()
    
    print("\n=== Key Insights ===")
    print("1. Normal equation gives exact solution (no iterations needed)")
    print("2. Requires matrix inversion - O(n³) complexity")
    print("3. Can be unstable if X^T X is close to singular")
    print("4. Works well for small to medium datasets")
    print("5. For large datasets, gradient descent is preferred")