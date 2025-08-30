"""
Logistic Regression Implementation from Scratch
Step 3: Probability & Statistics - Project 1

This module implements logistic regression from first principles, demonstrating:
- Sigmoid function and its probabilistic interpretation
- Maximum likelihood estimation and cross-entropy loss
- Gradient derivation and optimization
- Binary and multiclass classification

Mathematical Foundation:
P(y=1|x) = σ(w^T x + b) = 1/(1 + exp(-(w^T x + b)))
Loss: L = -Σ[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient: ∂L/∂w = X^T (ŷ - y)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    
    Supports both binary and multiclass classification using
    one-vs-rest approach for multiclass.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, regularization: Optional[str] = None,
                 lambda_reg: float = 0.01, random_seed: Optional[int] = None):
        """
        Initialize Logistic Regression.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence threshold for loss change
            regularization: Type of regularization ('l1', 'l2', or None)
            lambda_reg: Regularization strength
            random_seed: Random seed for reproducible results
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'weights_norm': []
        }
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid outputs
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax function for multiclass classification.
        
        Args:
            z: Input logits of shape (n_samples, n_classes)
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     weights: np.ndarray) -> float:
        """
        Compute cross-entropy loss with optional regularization.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            weights: Model weights (excluding bias)
            
        Returns:
            Total loss value
        """
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            # Binary classification
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multiclass classification
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Add regularization
        if self.regularization == 'l1':
            loss += self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            loss += self.lambda_reg * np.sum(weights ** 2) / 2
        
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients using the chain rule.
        
        Args:
            X: Feature matrix with bias column
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Gradients for weights and bias
        """
        n_samples = X.shape[0]
        
        # Gradient of cross-entropy loss
        dL_dz = y_pred - y_true
        
        # Gradients for weights and bias
        gradients = X.T @ dL_dz / n_samples
        
        # Add regularization gradients (only for weights, not bias)
        if self.regularization == 'l1':
            weight_gradients = gradients[1:]  # Exclude bias
            weight_gradients += self.lambda_reg * np.sign(self.weights)
            gradients[1:] = weight_gradients
        elif self.regularization == 'l2':
            weight_gradients = gradients[1:]  # Exclude bias
            weight_gradients += self.lambda_reg * self.weights
            gradients[1:] = weight_gradients
        
        return gradients[1:], gradients[0]  # weights, bias
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LogisticRegression':
        """
        Fit logistic regression using gradient descent.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels
            verbose: Whether to print training progress
            
        Returns:
            Fitted model
        """
        n_samples, n_features = X.shape
        
        # Handle multiclass labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            y_encoded = (y == self.classes_[1]).astype(int)
            n_outputs = 1
        else:
            # Multiclass classification - one-hot encoding
            y_encoded = np.zeros((n_samples, n_classes))
            for i, class_label in enumerate(self.classes_):
                y_encoded[:, i] = (y == class_label).astype(int)
            n_outputs = n_classes
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, (n_features, n_outputs))
        self.bias = np.zeros(n_outputs)
        
        # Add bias column to X
        X_with_bias = self._add_bias(X)
        
        # Training loop
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            
            if n_outputs == 1:
                # Binary classification
                y_pred = self._sigmoid(z).flatten()
                y_pred_reshaped = y_pred
                y_true_reshaped = y_encoded
            else:
                # Multiclass classification
                y_pred = self._softmax(z)
                y_pred_reshaped = y_pred
                y_true_reshaped = y_encoded
            
            # Compute loss
            current_loss = self._compute_loss(y_true_reshaped, y_pred_reshaped, self.weights)
            
            # Compute gradients
            weight_gradients, bias_gradient = self._compute_gradients(
                X_with_bias, y_true_reshaped, y_pred_reshaped
            )
            
            # Update parameters
            self.weights -= self.learning_rate * weight_gradients
            self.bias -= self.learning_rate * bias_gradient
            
            # Store history
            self.history['loss'].append(current_loss)
            
            # Compute accuracy
            if n_outputs == 1:
                predictions = (y_pred > 0.5).astype(int)
                accuracy = np.mean(predictions == y_encoded)
            else:
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y_encoded, axis=1)
                accuracy = np.mean(predictions == true_labels)
            
            self.history['accuracy'].append(accuracy)
            self.history['weights_norm'].append(np.linalg.norm(self.weights))
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_loss = current_loss
            
            # Print progress
            if verbose and (iteration + 1) % (self.max_iterations // 10) == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}: "
                      f"Loss = {current_loss:.6f}, Accuracy = {accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Test features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        z = X @ self.weights + self.bias
        
        if len(self.classes_) == 2:
            # Binary classification
            prob_positive = self._sigmoid(z).flatten()
            return np.column_stack([1 - prob_positive, prob_positive])
        else:
            # Multiclass classification
            return self._softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Test features
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demonstrate_binary_classification():
    """Demonstrate logistic regression on binary classification."""
    print("=== Binary Classification Demonstration ===")
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000, random_seed=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training curves
    axes[0].plot(model.history['loss'], label='Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curve
    axes[1].plot(model.history['accuracy'], label='Accuracy', color='green')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)[:, 1]  # Probability of class 1
    Z = Z.reshape(xx.shape)
    
    axes[2].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                             cmap='RdYlBu', edgecolors='black')
    axes[2].set_title(f'Decision Boundary\\nTest Accuracy: {test_accuracy:.3f}')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('logistic_regression_binary.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_multiclass_classification():
    """Demonstrate logistic regression on multiclass classification."""
    print("\\n=== Multiclass Classification Demonstration ===")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000, random_seed=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Confusion matrix-like analysis
    print("\\nPer-class performance:")
    for i, class_name in enumerate(target_names):
        class_mask = y_test == i
        if np.any(class_mask):
            class_accuracy = np.mean(y_pred[class_mask] == y_test[class_mask])
            avg_confidence = np.mean(y_proba[class_mask, i])
            print(f"  {class_name}: Accuracy = {class_accuracy:.3f}, "
                  f"Avg Confidence = {avg_confidence:.3f}")
    
    # Visualize results (using first 2 features for 2D plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training curves
    axes[0, 0].plot(model.history['loss'], label='Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curve
    axes[0, 1].plot(model.history['accuracy'], label='Accuracy', color='green')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary (first 2 features)
    X_2d = X_scaled[:, :2]  # Use first 2 features
    model_2d = LogisticRegression(learning_rate=0.1, max_iterations=1000, random_seed=42)
    model_2d.fit(X_2d[:-30], y[:-30])  # Train on subset for visualization
    
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    axes[1, 0].contourf(xx, yy, Z, alpha=0.8, cmap='Set3')
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        mask = y == i
        axes[1, 0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, 
                          label=target_names[i], edgecolors='black', alpha=0.7)
    
    axes[1, 0].set_title('Decision Boundary\\n(First 2 Features)')
    axes[1, 0].set_xlabel(feature_names[0])
    axes[1, 0].set_ylabel(feature_names[1])
    axes[1, 0].legend()
    
    # Plot 4: Prediction confidence
    confidence_scores = np.max(y_proba, axis=1)
    correct_predictions = (y_pred == y_test)
    
    axes[1, 1].hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
                   label='Correct', color='green', density=True)
    axes[1, 1].hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
                   label='Incorrect', color='red', density=True)
    axes[1, 1].set_xlabel('Prediction Confidence')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Confidence Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_multiclass.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_regularization():
    """Demonstrate effect of regularization on logistic regression."""
    print("\\n=== Regularization Effects ===")
    
    # Generate data with many features (some irrelevant)
    X, y = make_classification(n_samples=200, n_features=20, n_informative=5,
                             n_redundant=5, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Test different regularization strengths
    regularizations = [
        (None, 0, 'No Regularization'),
        ('l2', 0.01, 'L2 (λ=0.01)'),
        ('l2', 0.1, 'L2 (λ=0.1)'),
        ('l1', 0.01, 'L1 (λ=0.01)'),
        ('l1', 0.1, 'L1 (λ=0.1)')
    ]
    
    results = []
    
    for reg_type, lambda_reg, name in regularizations:
        model = LogisticRegression(
            learning_rate=0.1, max_iterations=1000,
            regularization=reg_type, lambda_reg=lambda_reg,
            random_seed=42
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        weight_norm = np.linalg.norm(model.weights)
        sparsity = np.mean(np.abs(model.weights) < 1e-3)  # Fraction of near-zero weights
        
        results.append({
            'name': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'weight_norm': weight_norm,
            'sparsity': sparsity,
            'weights': model.weights.flatten(),
            'history': model.history
        })
        
        print(f"{name:15s}: Train={train_acc:.3f}, Test={test_acc:.3f}, "
              f"||w||={weight_norm:.3f}, Sparsity={sparsity:.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy comparison
    names = [r['name'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, train_accs, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x_pos + width/2, test_accs, width, label='Test', alpha=0.8)
    axes[0, 0].set_xlabel('Regularization')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Weight norms
    weight_norms = [r['weight_norm'] for r in results]
    axes[0, 1].bar(names, weight_norms, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Regularization')
    axes[0, 1].set_ylabel('Weight Norm')
    axes[0, 1].set_title('Weight Magnitude')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sparsity
    sparsities = [r['sparsity'] for r in results]
    axes[0, 2].bar(names, sparsities, alpha=0.8, color='purple')
    axes[0, 2].set_xlabel('Regularization')
    axes[0, 2].set_ylabel('Fraction of Near-Zero Weights')
    axes[0, 2].set_title('Weight Sparsity')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4-6: Weight distributions
    for i, result in enumerate(results[:3]):  # Show first 3 for clarity
        ax = axes[1, i]
        weights = result['weights']
        ax.hist(weights, bins=30, alpha=0.7, density=True)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Weight Distribution\\n{result["name"]}')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_regularization.png', dpi=150, bbox_inches='tight')
    plt.show()


def sigmoid_analysis():
    """Analyze the sigmoid function and its properties."""
    print("\\n=== Sigmoid Function Analysis ===")
    
    # Generate range of inputs
    z = np.linspace(-10, 10, 1000)
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    
    # Plot sigmoid and its derivative
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Sigmoid function
    axes[0].plot(z, sigmoid, 'b-', linewidth=2, label='σ(z)')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision boundary')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('z')
    axes[0].set_ylabel('σ(z)')
    axes[0].set_title('Sigmoid Function')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sigmoid derivative
    axes[1].plot(z, sigmoid_derivative, 'g-', linewidth=2, label="σ'(z)")
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('z')
    axes[1].set_ylabel("σ'(z)")
    axes[1].set_title('Sigmoid Derivative')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Probability interpretation
    # Show how different z values correspond to confidence levels
    z_examples = [-3, -1, 0, 1, 3]
    probs = 1 / (1 + np.exp(-np.array(z_examples)))
    
    axes[2].bar(range(len(z_examples)), probs, alpha=0.7, color='purple')
    axes[2].set_xlabel('Input z')
    axes[2].set_ylabel('P(y=1|x)')
    axes[2].set_title('Probabilistic Interpretation')
    axes[2].set_xticks(range(len(z_examples)))
    axes[2].set_xticklabels([f'z={z}' for z in z_examples])
    axes[2].grid(True, alpha=0.3)
    
    # Add probability values as text
    for i, (z_val, prob) in enumerate(zip(z_examples, probs)):
        axes[2].text(i, prob + 0.05, f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sigmoid_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Key Properties of Sigmoid Function:")
    print("1. Maps any real number to (0, 1) - perfect for probabilities")
    print("2. S-shaped curve with smooth transitions")
    print("3. Maximum derivative at z=0 (σ'(0) = 0.25)")
    print("4. Approaches 0 for large negative z, 1 for large positive z")
    print("5. Symmetric around z=0 where σ(0) = 0.5")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_binary_classification()
    demonstrate_multiclass_classification()
    demonstrate_regularization()
    sigmoid_analysis()
    
    print("\\n=== Key Insights ===")
    print("1. Logistic regression models probability using sigmoid function")
    print("2. Maximum likelihood estimation leads to cross-entropy loss")
    print("3. Gradient has elegant form: (prediction - target)")
    print("4. Regularization prevents overfitting and creates sparse models")
    print("5. Probabilistic interpretation enables uncertainty quantification")