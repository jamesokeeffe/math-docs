"""
2-Layer Neural Network Implementation from Scratch
Step 2: Calculus & Gradients - Project 2

This module implements a complete 2-layer neural network with:
- Forward propagation using matrix operations
- Backward propagation using chain rule
- Multiple activation functions and their derivatives
- Training loop with gradient descent
- Visualization of learning process

Mathematical Foundation:
Forward: z₁ = W₁x + b₁, a₁ = σ(z₁), z₂ = W₂a₁ + b₂, ŷ = σ(z₂)
Backward: Use chain rule to compute ∂L/∂W₁, ∂L/∂W₂, ∂L/∂b₁, ∂L/∂b₂
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ActivationFunction:
    """Base class for activation functions and their derivatives."""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(x)
        return s * (1 - s)

class ReLU(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Tanh(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

class LossFunction:
    """Base class for loss functions and their derivatives."""
    
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.5 * np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / len(y_true)

class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)

class TwoLayerNeuralNetwork:
    """
    A 2-layer neural network implemented from scratch.
    
    Architecture: Input → Hidden Layer → Output Layer
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hidden_activation: ActivationFunction = ReLU,
                 output_activation: ActivationFunction = Sigmoid,
                 loss_function: LossFunction = BinaryCrossEntropy,
                 learning_rate: float = 0.01,
                 random_seed: Optional[int] = None):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
            hidden_activation: Activation function for hidden layer
            output_activation: Activation function for output layer
            loss_function: Loss function to optimize
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducible initialization
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Activation functions
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        
        # Initialize weights and biases using Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'weights_norm': []
        }
    
    def forward(self, X: np.ndarray, save_cache: bool = True) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, input_size)
            save_cache: Whether to save intermediate values for backpropagation
            
        Returns:
            Output predictions of shape (n_samples, output_size)
        """
        # Layer 1: Input → Hidden
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.hidden_activation.forward(self.z1)
        
        # Layer 2: Hidden → Output
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.output_activation.forward(self.z2)
        
        if save_cache:
            self.X_cache = X
        
        return self.a2
    
    def backward(self, y_true: np.ndarray) -> dict:
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels of shape (n_samples, output_size)
            
        Returns:
            Dictionary containing all gradients
        """
        m = y_true.shape[0]  # Number of samples
        
        # Output layer gradients
        dL_da2 = self.loss_function.backward(y_true, self.a2)
        da2_dz2 = self.output_activation.backward(self.z2)
        dL_dz2 = dL_da2 * da2_dz2
        
        # Gradients for W2 and b2
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients (backpropagate through W2)
        dL_da1 = dL_dz2 @ self.W2.T
        da1_dz1 = self.hidden_activation.backward(self.z1)
        dL_dz1 = dL_da1 * da1_dz1
        
        # Gradients for W1 and b1
        dL_dW1 = self.X_cache.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        return {
            'dW1': dL_dW1,
            'db1': dL_db1,
            'dW2': dL_dW2,
            'db2': dL_db2
        }
    
    def update_weights(self, gradients: dict):
        """Update weights using gradient descent."""
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss between true and predicted values."""
        return self.loss_function.forward(y_true, y_pred)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        if self.output_size == 1:
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions.flatten() == y_true.flatten())
        else:
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              batch_size: Optional[int] = None, verbose: bool = True,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, input_size)
            y: Training labels of shape (n_samples, output_size)
            epochs: Number of training epochs
            batch_size: Mini-batch size (None for full batch)
            verbose: Whether to print training progress
            validation_data: Tuple of (X_val, y_val) for validation
            
        Returns:
            Training history
        """
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss and accuracy
                batch_loss = self.compute_loss(y_batch, y_pred)
                batch_accuracy = self.compute_accuracy(y_batch, y_pred)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                n_batches += 1
                
                # Backward pass
                gradients = self.backward(y_batch)
                
                # Update weights
                self.update_weights(gradients)
            
            # Average metrics over batches
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            # Store history
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            
            # Compute weight norms for analysis
            w1_norm = np.linalg.norm(self.W1)
            w2_norm = np.linalg.norm(self.W2)
            self.history['weights_norm'].append(w1_norm + w2_norm)
            
            # Validation metrics
            val_loss, val_accuracy = None, None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val, save_cache=False)
                val_loss = self.compute_loss(y_val, y_val_pred)
                val_accuracy = self.compute_accuracy(y_val, y_val_pred)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                val_str = ""
                if validation_data is not None:
                    val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f}{val_str}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.forward(X, save_cache=False)
    
    def visualize_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                                  title: str = "Decision Boundary", 
                                  resolution: int = 100):
        """Visualize decision boundary for 2D data."""
        if X.shape[1] != 2:
            print("Decision boundary visualization only works for 2D data")
            return
        
        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), 
                            cmap='RdYlBu', edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()


def demonstrate_binary_classification():
    """Demonstrate neural network on binary classification problem."""
    print("=== Binary Classification Demonstration ===")
    
    # Generate datasets
    datasets = {
        'Linear': make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                    n_informative=2, random_state=42, n_clusters_per_class=1),
        'Moons': make_moons(n_samples=300, noise=0.3, random_state=42),
        'Circles': make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, (X, y)) in enumerate(datasets.items()):
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_reshaped = y.reshape(-1, 1)
        
        # Create and train neural network
        nn = TwoLayerNeuralNetwork(
            input_size=2, hidden_size=10, output_size=1,
            hidden_activation=ReLU, output_activation=Sigmoid,
            learning_rate=0.1, random_seed=42
        )
        
        # Train the network
        history = nn.train(X_scaled, y_reshaped, epochs=1000, verbose=False)
        
        # Plot original data
        axes[0, idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdYlBu')
        axes[0, idx].set_title(f'{name} Dataset')
        axes[0, idx].set_xlabel('Feature 1')
        axes[0, idx].set_ylabel('Feature 2')
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        axes[1, idx].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        axes[1, idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                           cmap='RdYlBu', edgecolors='black')
        
        final_accuracy = history['accuracy'][-1]
        axes[1, idx].set_title(f'{name} Decision Boundary\\nAccuracy: {final_accuracy:.3f}')
        axes[1, idx].set_xlabel('Feature 1')
        axes[1, idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('binary_classification_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_training_dynamics():
    """Analyze how neural networks learn during training."""
    print("\\n=== Training Dynamics Analysis ===")
    
    # Generate complex dataset
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = y.reshape(-1, 1)
    
    # Split into train/validation
    split = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_reshaped[:split], y_reshaped[split:]
    
    # Train network with detailed tracking
    nn = TwoLayerNeuralNetwork(
        input_size=2, hidden_size=20, output_size=1,
        learning_rate=0.1, random_seed=42
    )
    
    print("Training neural network...")
    history = nn.train(X_train, y_train, epochs=2000, 
                      validation_data=(X_val, y_val), verbose=True)
    
    # Plot training dynamics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['accuracy'], 'g-', label='Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight norms
    axes[0, 2].plot(epochs, history['weights_norm'], 'r-')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Weight Norm')
    axes[0, 2].set_title('Weight Magnitudes')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Weight distributions at different epochs
    epoch_snapshots = [0, len(epochs)//4, len(epochs)//2, len(epochs)-1]
    
    # We'll need to retrain to capture weight snapshots
    weight_snapshots = []
    nn_snapshot = TwoLayerNeuralNetwork(input_size=2, hidden_size=20, output_size=1,
                                      learning_rate=0.1, random_seed=42)
    
    for i, epoch in enumerate(epoch_snapshots):
        if epoch == 0:
            weights = np.concatenate([nn_snapshot.W1.flatten(), nn_snapshot.W2.flatten()])
        else:
            # Train up to this epoch
            nn_snapshot.train(X_train, y_train, epochs=epoch_snapshots[i] - (epoch_snapshots[i-1] if i > 0 else 0), 
                            verbose=False)
            weights = np.concatenate([nn_snapshot.W1.flatten(), nn_snapshot.W2.flatten()])
        weight_snapshots.append(weights)
    
    # Plot weight histograms
    axes[1, 0].hist(weight_snapshots[0], bins=30, alpha=0.5, label='Initial', density=True)
    axes[1, 0].hist(weight_snapshots[-1], bins=30, alpha=0.5, label='Final', density=True)
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Weight Distribution Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hidden layer activations
    hidden_activations = nn.a1
    axes[1, 1].hist(hidden_activations.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Activation Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Hidden Layer Activations')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final decision boundary
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    axes[1, 2].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    axes[1, 2].scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), 
                      cmap='RdYlBu', edgecolors='black', alpha=0.7, label='Train')
    axes[1, 2].scatter(X_val[:, 0], X_val[:, 1], c=y_val.flatten(), 
                      cmap='RdYlBu', edgecolors='white', s=100, marker='^', label='Val')
    axes[1, 2].set_title('Final Decision Boundary')
    axes[1, 2].set_xlabel('Feature 1')
    axes[1, 2].set_ylabel('Feature 2')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_activation_functions():
    """Compare different activation functions."""
    print("\\n=== Activation Function Comparison ===")
    
    # Generate dataset
    X, y = make_circles(n_samples=400, noise=0.1, factor=0.3, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_reshaped = y.reshape(-1, 1)
    
    # Test different activation functions
    activations = {
        'ReLU': ReLU,
        'Sigmoid': Sigmoid,
        'Tanh': Tanh
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, activation) in enumerate(activations.items()):
        # Train network
        nn = TwoLayerNeuralNetwork(
            input_size=2, hidden_size=15, output_size=1,
            hidden_activation=activation, output_activation=Sigmoid,
            learning_rate=0.1, random_seed=42
        )
        
        history = nn.train(X_scaled, y_reshaped, epochs=1000, verbose=False)
        
        # Plot loss curve
        axes[0, idx].plot(history['loss'])
        axes[0, idx].set_title(f'{name} - Training Loss')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Loss')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        axes[1, idx].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        axes[1, idx].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, 
                           cmap='RdYlBu', edgecolors='black')
        
        final_accuracy = history['accuracy'][-1]
        axes[1, idx].set_title(f'{name} - Decision Boundary\\nAcc: {final_accuracy:.3f}')
        axes[1, idx].set_xlabel('Feature 1')
        axes[1, idx].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_binary_classification()
    analyze_training_dynamics()
    demonstrate_activation_functions()
    
    print("\\n=== Key Insights ===")
    print("1. Forward pass computes outputs through matrix operations")
    print("2. Backward pass uses chain rule to compute gradients efficiently")
    print("3. Different activation functions create different decision boundaries")
    print("4. Learning rate and architecture affect convergence speed")
    print("5. Neural networks can learn complex non-linear patterns")