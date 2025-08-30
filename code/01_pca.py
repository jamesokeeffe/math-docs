"""
Principal Component Analysis (PCA) Implementation from Scratch
Step 1: Linear Algebra - Project 2

Mathematical Steps:
1. Center the data: X_centered = X - mean(X)
2. Compute covariance matrix: C = (1/n) X_centered^T X_centered
3. Find eigenvalues and eigenvectors of C
4. Sort eigenvalues in descending order
5. Select top k eigenvectors as principal components
6. Transform data: X_transformed = X_centered @ principal_components

This implementation demonstrates:
- Eigenvalue decomposition in practice
- Dimensionality reduction
- Variance explanation and data compression
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler

class PCA:
    """
    Principal Component Analysis implementation from scratch.
    
    PCA finds the directions (principal components) along which the data
    varies the most. It's a fundamental technique for dimensionality reduction
    and data visualization.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep. If None, keep all.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA on the data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            self: Fitted PCA object
        """
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Step 3: Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort in descending order
        # eigh returns eigenvalues in ascending order, so reverse
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select top k components
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Singular values (for compatibility with sklearn)
        self.singular_values_ = np.sqrt(self.explained_variance_ * (n_samples - 1))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Data matrix
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Data in PC space
            
        Returns:
            Reconstructed data in original space
        """
        return X_transformed @ self.components_ + self.mean_


def demonstrate_pca_2d():
    """
    Demonstrate PCA on 2D data for clear visualization.
    """
    print("=== PCA Demonstration on 2D Data ===\n")
    
    # Generate correlated 2D data
    np.random.seed(42)
    n_samples = 200
    
    # Create data with clear correlation
    angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    
    # Generate data with different variances along axes
    data_original = np.random.multivariate_normal([0, 0], [[3, 0], [0, 1]], n_samples)
    data = data_original @ rotation_matrix.T  # Rotate the data
    
    # Apply PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    print(f"Original data shape: {data.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original data
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # Plot 2: Data with principal components
    axes[0, 1].scatter(data[:, 0], data[:, 1], alpha=0.6)
    
    # Draw principal components
    mean_point = pca.mean_
    for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
        # Scale arrows by square root of eigenvalue for visibility
        scale = 3 * np.sqrt(variance)
        axes[0, 1].arrow(mean_point[0], mean_point[1], 
                        scale * component[0], scale * component[1],
                        head_width=0.1, head_length=0.1, 
                        fc=f'C{i}', ec=f'C{i}', width=0.02,
                        label=f'PC{i+1} (var={variance:.2f})')
    
    axes[0, 1].set_title('Data with Principal Components')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    
    # Plot 3: Transformed data (PC space)
    axes[1, 0].scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6)
    axes[1, 0].set_title('Data in PC Space')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[1, 0].grid(True)
    axes[1, 0].axis('equal')
    
    # Plot 4: Reconstruction with 1 component
    pca_1d = PCA(n_components=1)
    data_1d = pca_1d.fit_transform(data)
    data_reconstructed = pca_1d.inverse_transform(data_1d)
    
    axes[1, 1].scatter(data[:, 0], data[:, 1], alpha=0.4, label='Original', s=20)
    axes[1, 1].scatter(data_reconstructed[:, 0], data_reconstructed[:, 1], 
                      alpha=0.8, label='Reconstructed (1 PC)', s=20)
    
    # Draw lines showing the projection
    for i in range(0, len(data), 10):  # Every 10th point for clarity
        axes[1, 1].plot([data[i, 0], data_reconstructed[i, 0]], 
                       [data[i, 1], data_reconstructed[i, 1]], 
                       'k-', alpha=0.3, linewidth=0.5)
    
    axes[1, 1].set_title(f'1D Reconstruction ({pca_1d.explained_variance_ratio_[0]:.1%} variance retained)')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('pca_2d_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_pca_iris():
    """
    Demonstrate PCA on the famous Iris dataset.
    """
    print("\n=== PCA on Iris Dataset ===\n")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Original data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nExplained variance ratio by component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\nCumulative explained variance:")
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i, cum_ratio in enumerate(cumsum):
        print(f"  PC1-{i+1}: {cum_ratio:.3f} ({cum_ratio*100:.1f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Explained variance
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
    axes[0, 0].plot(range(1, len(cumsum) + 1), cumsum, 'ro-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('PCA Explained Variance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(['Cumulative', 'Individual'])
    
    # Plot 2: PC1 vs PC2
    colors = ['red', 'green', 'blue']
    for i, (target, color) in enumerate(zip(target_names, colors)):
        mask = y == i
        axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=color, label=target, alpha=0.7)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 1].set_title('Iris Data in PC Space (PC1 vs PC2)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PC1 vs PC3
    for i, (target, color) in enumerate(zip(target_names, colors)):
        mask = y == i
        axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 2], 
                          c=color, label=target, alpha=0.7)
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1, 0].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
    axes[1, 0].set_title('Iris Data in PC Space (PC1 vs PC3)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Component contributions (loadings)
    components_df = pca.components_[:3].T  # First 3 components
    
    im = axes[1, 1].imshow(components_df, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels(['PC1', 'PC2', 'PC3'])
    axes[1, 1].set_yticks(range(len(feature_names)))
    axes[1, 1].set_yticklabels([name.replace(' (cm)', '') for name in feature_names])
    axes[1, 1].set_title('Principal Component Loadings')
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(3):
            text = axes[1, 1].text(j, i, f'{components_df[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('pca_iris_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_pca_compression():
    """
    Demonstrate PCA for data compression and reconstruction.
    """
    print("\n=== PCA for Data Compression ===\n")
    
    # Generate high-dimensional data with intrinsic lower dimensionality
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create data that lies approximately on a 5D manifold
    true_dim = 5
    latent_data = np.random.randn(n_samples, true_dim)
    
    # Random projection to high-dimensional space
    projection_matrix = np.random.randn(true_dim, n_features)
    X = latent_data @ projection_matrix
    
    # Add some noise
    X += 0.1 * np.random.randn(n_samples, n_features)
    
    print(f"Original data shape: {X.shape}")
    
    # Apply PCA with different numbers of components
    reconstruction_errors = []
    compression_ratios = []
    components_range = range(1, min(21, n_features))
    
    for n_comp in components_range:
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        mse = np.mean((X - X_reconstructed) ** 2)
        reconstruction_errors.append(mse)
        
        # Calculate compression ratio
        original_size = n_samples * n_features
        compressed_size = n_samples * n_comp + n_comp * n_features  # Data + components
        compression_ratio = original_size / compressed_size
        compression_ratios.append(compression_ratio)
        
        if n_comp in [1, 5, 10, 15]:
            print(f"Components: {n_comp:2d}, "
                  f"Variance explained: {np.sum(pca.explained_variance_ratio_):.3f}, "
                  f"Reconstruction MSE: {mse:.6f}, "
                  f"Compression ratio: {compression_ratio:.1f}x")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Reconstruction error vs number of components
    ax1.plot(components_range, reconstruction_errors, 'bo-', linewidth=2)
    ax1.axvline(x=true_dim, color='red', linestyle='--', 
                label=f'True dimensionality ({true_dim})')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Reconstruction MSE')
    ax1.set_title('Reconstruction Error vs Components')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Compression ratio vs number of components
    ax2.plot(components_range, compression_ratios, 'go-', linewidth=2)
    ax2.axvline(x=true_dim, color='red', linestyle='--', 
                label=f'True dimensionality ({true_dim})')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression Ratio vs Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('pca_compression_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def mathematical_deep_dive():
    """
    Deep dive into the mathematical details of PCA.
    """
    print("\n=== Mathematical Deep Dive ===\n")
    
    # Simple 3x2 example for manual calculation
    X = np.array([[1, 2],
                  [3, 4], 
                  [5, 6]])
    
    print("Original data matrix X:")
    print(X)
    
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    print(f"\nMean: {mean}")
    print("Centered data X_centered:")
    print(X_centered)
    
    # Step 2: Covariance matrix
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    print(f"\nCovariance matrix C = X_centered^T @ X_centered / (n-1):")
    print(cov_matrix)
    
    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print(f"\nEigenvalues: {eigenvalues}")
    print("Eigenvectors:")
    print(eigenvectors)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    print(f"\nSorted eigenvalues: {eigenvalues_sorted}")
    print("Sorted eigenvectors (columns are eigenvectors):")
    print(eigenvectors_sorted)
    
    # Step 4: Verify eigenvalue equation Av = λv
    print("\n--- Verification: Av = λv ---")
    for i, (val, vec) in enumerate(zip(eigenvalues_sorted, eigenvectors_sorted.T)):
        Av = cov_matrix @ vec
        lambda_v = val * vec
        print(f"\nEigenvector {i+1}:")
        print(f"  Av = {Av}")
        print(f"  λv = {lambda_v}")
        print(f"  Equal? {np.allclose(Av, lambda_v)}")
    
    # Step 5: Transform data
    principal_components = eigenvectors_sorted.T  # Rows are PCs
    X_transformed = X_centered @ principal_components.T
    print(f"\nTransformed data (PC space):")
    print(X_transformed)
    
    # Step 6: Verify variance in PC space
    pc_variances = np.var(X_transformed, axis=0, ddof=1)
    print(f"\nVariances in PC space: {pc_variances}")
    print(f"Eigenvalues:           {eigenvalues_sorted}")
    print(f"Equal? {np.allclose(pc_variances, eigenvalues_sorted)}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_pca_2d()
    demonstrate_pca_iris()
    demonstrate_pca_compression()
    mathematical_deep_dive()
    
    print("\n=== Key Insights ===")
    print("1. PCA finds orthogonal directions of maximum variance")
    print("2. Principal components are eigenvectors of covariance matrix")
    print("3. Eigenvalues represent variance explained by each component")
    print("4. Dimensionality reduction preserves most important information")
    print("5. Useful for data compression, visualization, and noise reduction")