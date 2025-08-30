"""
Gradient Descent Implementation and Visualization
Step 2: Calculus & Gradients - Project 1

This module demonstrates gradient descent optimization on various functions,
showing how calculus enables automatic optimization.

Mathematical Foundation:
For function f(x), gradient descent updates: x_{t+1} = x_t - α∇f(x_t)
where α is the learning rate and ∇f is the gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Optional
import warnings

class GradientDescent:
    """
    Gradient Descent optimizer for scalar and vector functions.
    
    Supports various learning rate schedules and convergence criteria.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, momentum: float = 0.0):
        """
        Initialize gradient descent optimizer.
        
        Args:
            learning_rate: Step size for gradient updates
            max_iterations: Maximum number of optimization steps
            tolerance: Convergence threshold for gradient magnitude
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.momentum = momentum
        
        # Track optimization history
        self.history = {
            'x': [],
            'f_values': [],
            'gradients': [],
            'step_sizes': []
        }
    
    def optimize(self, f: Callable, gradient_f: Callable, x0: np.ndarray,
                verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Optimize function f starting from x0.
        
        Args:
            f: Function to minimize
            gradient_f: Gradient of the function
            x0: Starting point
            verbose: Whether to print progress
            
        Returns:
            Optimal point and function value
        """
        x = np.array(x0, dtype=float)
        velocity = np.zeros_like(x)  # For momentum
        
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
        for iteration in range(self.max_iterations):
            # Compute function value and gradient
            f_val = f(x)
            grad = gradient_f(x)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f_values'].append(f_val)
            self.history['gradients'].append(grad.copy())
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration} iterations. Gradient norm: {grad_norm:.2e}")
                break
            
            # Update with momentum
            velocity = self.momentum * velocity - self.learning_rate * grad
            step_size = np.linalg.norm(velocity)
            self.history['step_sizes'].append(step_size)
            
            x = x + velocity
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: f(x)={f_val:.6f}, ||∇f||={grad_norm:.6f}")
        
        return x, f(x)


def quadratic_1d(x: float) -> float:
    """Simple quadratic function: f(x) = (x-2)² + 1"""
    return (x - 2)**2 + 1

def quadratic_1d_grad(x: float) -> float:
    """Gradient of quadratic function: f'(x) = 2(x-2)"""
    return 2 * (x - 2)

def quadratic_2d(x: np.ndarray) -> float:
    """2D quadratic: f(x,y) = (x-1)² + 2(y+1)² + 3"""
    return (x[0] - 1)**2 + 2 * (x[1] + 1)**2 + 3

def quadratic_2d_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of 2D quadratic: [2(x-1), 4(y+1)]"""
    return np.array([2 * (x[0] - 1), 4 * (x[1] + 1)])

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = 100(y-x²)² + (1-x)²"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    df_dx = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dy = 200 * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

def beale(x: np.ndarray) -> float:
    """Beale function: challenging optimization problem"""
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3

def beale_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Beale function"""
    # Compute partial derivatives
    t1 = 1.5 - x[0] + x[0]*x[1]
    t2 = 2.25 - x[0] + x[0]*x[1]**2
    t3 = 2.625 - x[0] + x[0]*x[1]**3
    
    df_dx = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
    df_dy = 2*t1*x[0] + 2*t2*x[0]*2*x[1] + 2*t3*x[0]*3*x[1]**2
    
    return np.array([df_dx, df_dy])


def demonstrate_1d_optimization():
    """Demonstrate gradient descent on 1D quadratic function."""
    print("=== 1D Quadratic Function Optimization ===")
    print("Function: f(x) = (x-2)² + 1")
    print("Gradient: f'(x) = 2(x-2)")
    print("True minimum: x = 2, f(2) = 1")
    
    # Test different learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    starting_points = [-5, 0, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        
        # Plot function
        x_range = np.linspace(-6, 8, 1000)
        y_range = [quadratic_1d(x) for x in x_range]
        ax.plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
        
        # Test different starting points
        colors = ['red', 'green', 'orange']
        for start, color in zip(starting_points, colors):
            optimizer = GradientDescent(learning_rate=lr, max_iterations=50)
            x_opt, f_opt = optimizer.optimize(quadratic_1d, quadratic_1d_grad, 
                                            np.array([start]))
            
            # Plot optimization path
            x_history = [x[0] for x in optimizer.history['x']]
            f_history = optimizer.history['f_values']
            
            ax.plot(x_history, f_history, 'o-', color=color, alpha=0.7,
                   markersize=4, linewidth=1, 
                   label=f'Start: {start} → {x_opt[0]:.2f}')
            
            # Mark starting and ending points
            ax.plot(start, quadratic_1d(start), 'o', color=color, markersize=8)
            ax.plot(x_opt[0], f_opt, 's', color=color, markersize=8)
        
        ax.set_title(f'Learning Rate = {lr}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 20)
    
    plt.suptitle('Effect of Learning Rate on Convergence')
    plt.tight_layout()
    plt.savefig('gradient_descent_1d.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_2d_optimization():
    """Demonstrate gradient descent on 2D functions with contour plots."""
    print("\n=== 2D Function Optimization ===")
    
    functions = [
        (quadratic_2d, quadratic_2d_grad, "Quadratic", [-3, 3], [2, 1]),
        (rosenbrock, rosenbrock_grad, "Rosenbrock", [-2, 2], [0, 0])
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (func, grad_func, name, start, optimal) in enumerate(functions):
        ax = axes[idx]
        
        # Create contour plot
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))\n        \n        # Plot contours\n        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)\n        ax.clabel(contour, inline=True, fontsize=8)\n        \n        # Optimize with different learning rates\n        learning_rates = [0.001, 0.01, 0.1]\n        colors = ['red', 'green', 'blue']\n        \n        for lr, color in zip(learning_rates, colors):\n            optimizer = GradientDescent(learning_rate=lr, max_iterations=200)\n            x_opt, f_opt = optimizer.optimize(func, grad_func, np.array(start))\n            \n            # Plot optimization path\n            x_history = np.array(optimizer.history['x'])\n            \n            ax.plot(x_history[:, 0], x_history[:, 1], 'o-', \n                   color=color, alpha=0.8, markersize=3, linewidth=1,\n                   label=f'LR={lr}: {len(x_history)} steps')\n            \n            # Mark start and end\n            ax.plot(start[0], start[1], 'o', color=color, markersize=8)\n            ax.plot(x_opt[0], x_opt[1], 's', color=color, markersize=8)\n        \n        # Mark true optimum\n        ax.plot(optimal[0], optimal[1], 'k*', markersize=15, label='True optimum')\n        \n        ax.set_title(f'{name} Function')\n        ax.set_xlabel('x')\n        ax.set_ylabel('y')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        ax.set_aspect('equal')\n    \n    plt.tight_layout()\n    plt.savefig('gradient_descent_2d.png', dpi=150, bbox_inches='tight')\n    plt.show()


def analyze_convergence():
    """Analyze convergence properties of gradient descent."""
    print("\n=== Convergence Analysis ===")
    
    # Test on simple quadratic with different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9, 1.1, 1.5]
    
    convergence_data = []
    
    for lr in learning_rates:
        optimizer = GradientDescent(learning_rate=lr, max_iterations=200, tolerance=1e-8)
        x_opt, f_opt = optimizer.optimize(quadratic_2d, quadratic_2d_grad, np.array([5, 5]))\n        \n        # Analyze convergence\n        f_history = np.array(optimizer.history['f_values'])\n        grad_norms = [np.linalg.norm(g) for g in optimizer.history['gradients']]\n        \n        convergence_data.append({\n            'lr': lr,\n            'iterations': len(f_history),\n            'final_f': f_history[-1],\n            'final_grad_norm': grad_norms[-1],\n            'f_history': f_history,\n            'grad_norms': grad_norms\n        })\n    \n    # Plot convergence curves\n    fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n    \n    # Plot 1: Function value vs iteration\n    for data in convergence_data:\n        if data['lr'] <= 1.0:  # Only stable learning rates\n            axes[0, 0].semilogy(data['f_history'], label=f"LR={data['lr']}")\n    \n    axes[0, 0].axhline(y=3, color='black', linestyle='--', label='True minimum')\n    axes[0, 0].set_xlabel('Iteration')\n    axes[0, 0].set_ylabel('Function Value (log scale)')\n    axes[0, 0].set_title('Function Value Convergence')\n    axes[0, 0].legend()\n    axes[0, 0].grid(True, alpha=0.3)\n    \n    # Plot 2: Gradient norm vs iteration\n    for data in convergence_data:\n        if data['lr'] <= 1.0:\n            axes[0, 1].semilogy(data['grad_norms'], label=f"LR={data['lr']}")\n    \n    axes[0, 1].set_xlabel('Iteration')\n    axes[0, 1].set_ylabel('Gradient Norm (log scale)')\n    axes[0, 1].set_title('Gradient Norm Convergence')\n    axes[0, 1].legend()\n    axes[0, 1].grid(True, alpha=0.3)\n    \n    # Plot 3: Learning rate vs final error\n    lrs = [data['lr'] for data in convergence_data]\n    final_errors = [abs(data['final_f'] - 3) for data in convergence_data]\n    \n    axes[1, 0].loglog(lrs, final_errors, 'bo-')\n    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', label='Stability threshold')\n    axes[1, 0].set_xlabel('Learning Rate')\n    axes[1, 0].set_ylabel('Final Error (log scale)')\n    axes[1, 0].set_title('Learning Rate vs Final Error')\n    axes[1, 0].legend()\n    axes[1, 0].grid(True, alpha=0.3)\n    \n    # Plot 4: Learning rate vs iterations to converge\n    iterations = [data['iterations'] for data in convergence_data]\n    \n    axes[1, 1].semilogx(lrs, iterations, 'ro-')\n    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', label='Stability threshold')\n    axes[1, 1].set_xlabel('Learning Rate')\n    axes[1, 1].set_ylabel('Iterations to Converge')\n    axes[1, 1].set_title('Learning Rate vs Convergence Speed')\n    axes[1, 1].legend()\n    axes[1, 1].grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')\n    plt.show()\n    \n    # Print analysis\n    print("Learning Rate Analysis:")\n    for data in convergence_data:\n        status = "STABLE" if data['lr'] <= 1.0 and data['final_grad_norm'] < 1e-3 else "UNSTABLE"\n        print(f"  LR={data['lr']:4.1f}: {data['iterations']:3d} iterations, "\n              f"final_error={abs(data['final_f']-3):.2e}, status={status}")


def demonstrate_momentum():
    """Demonstrate the effect of momentum on optimization."""
    print("\n=== Momentum Demonstration ===")
    
    # Use Rosenbrock function (has challenging narrow valley)\n    momentum_values = [0.0, 0.5, 0.9, 0.99]\n    \n    fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n    axes = axes.ravel()\n    \n    # Create contour plot data\n    x = np.linspace(-2, 2, 100)\n    y = np.linspace(-1, 3, 100)\n    X, Y = np.meshgrid(x, y)\n    Z = np.zeros_like(X)\n    \n    for i in range(X.shape[0]):\n        for j in range(X.shape[1]):\n            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))\n    \n    for idx, momentum in enumerate(momentum_values):\n        ax = axes[idx]\n        \n        # Plot contours\n        contour = ax.contour(X, Y, Z, levels=np.logspace(0, 3, 20), alpha=0.6)\n        \n        # Optimize with momentum\n        optimizer = GradientDescent(learning_rate=0.001, max_iterations=1000, \n                                  momentum=momentum)\n        x_opt, f_opt = optimizer.optimize(rosenbrock, rosenbrock_grad, \n                                        np.array([-1.5, 2.5]))\n        \n        # Plot optimization path\n        x_history = np.array(optimizer.history['x'])\n        \n        ax.plot(x_history[:, 0], x_history[:, 1], 'r-', alpha=0.8, linewidth=2)\n        ax.plot(x_history[::10, 0], x_history[::10, 1], 'ro', markersize=3)\n        \n        # Mark start and end\n        ax.plot(-1.5, 2.5, 'go', markersize=8, label='Start')\n        ax.plot(1, 1, 'k*', markersize=15, label='True optimum')\n        ax.plot(x_opt[0], x_opt[1], 'rs', markersize=8, label='Final')\n        \n        ax.set_title(f'Momentum = {momentum}\\n{len(x_history)} iterations')\n        ax.set_xlabel('x')\n        ax.set_ylabel('y')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        ax.set_xlim(-2, 2)\n        ax.set_ylim(-1, 3)\n    \n    plt.suptitle('Effect of Momentum on Rosenbrock Function')\n    plt.tight_layout()\n    plt.savefig('momentum_comparison.png', dpi=150, bbox_inches='tight')\n    plt.show()


if __name__ == "__main__":\n    # Run all demonstrations\n    demonstrate_1d_optimization()\n    demonstrate_2d_optimization()\n    analyze_convergence()\n    demonstrate_momentum()\n    \n    print("\n=== Key Insights ===")\n    print("1. Learning rate controls convergence speed vs stability")\n    print("2. Too high learning rate causes oscillation/divergence")\n    print("3. Too low learning rate causes slow convergence")\n    print("4. Momentum helps escape local minima and accelerates convergence")\n    print("5. Different functions have different optimization landscapes")