# Step 1: Linear Algebra – The Foundations

## Overview

Linear algebra is the mathematical foundation of machine learning. Every neural network operation, from basic matrix multiplication to complex transformations, relies on linear algebra concepts. In this step, you'll build intuitive understanding and practical skills.

## 🎯 Learning Objectives

By the end of this step, you will:
- Understand vectors and matrices as geometric and computational objects
- Master essential operations: dot products, matrix multiplication, transposes
- Compute eigenvalues, eigenvectors, and perform matrix diagonalization
- Implement Singular Value Decomposition (SVD) and understand its applications
- See how neural networks are fundamentally matrix operations
- Implement linear regression and PCA from scratch

## 📚 Mathematical Foundations

### Vectors: The Building Blocks

A vector is an ordered list of numbers that represents:
- **Geometrically**: A point in space or a direction and magnitude
- **Computationally**: A collection of features or data

```
v = [v₁, v₂, v₃, ..., vₙ]
```

**Key Properties:**
- **Magnitude (L2 norm)**: ||v|| = √(v₁² + v₂² + ... + vₙ²)
- **Unit vector**: v/||v|| (direction only, magnitude = 1)
- **Vector addition**: Element-wise addition
- **Scalar multiplication**: Multiply each element by a scalar

### Dot Product: Measuring Similarity

The dot product between vectors u and v:
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ = ||u|| ||v|| cos(θ)
```

**Intuition**: 
- Measures how much vectors point in the same direction
- When θ = 0°: cos(θ) = 1 (parallel vectors)
- When θ = 90°: cos(θ) = 0 (orthogonal vectors)
- When θ = 180°: cos(θ) = -1 (opposite vectors)

**ML Applications**:
- Similarity measures in recommendation systems
- Attention mechanisms in transformers
- Feature correlation analysis

### Matrices: Transformations and Data

A matrix is a rectangular array of numbers:
```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [...  ...  ...  ...]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

**Key Operations:**

1. **Matrix Multiplication** (AB):
   - (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ
   - Represents composition of transformations
   - Not commutative: AB ≠ BA (usually)

2. **Transpose** (Aᵀ):
   - Flip rows and columns
   - (Aᵀ)ᵢⱼ = Aⱼᵢ

3. **Inverse** (A⁻¹):
   - AA⁻¹ = A⁻¹A = I (identity matrix)
   - Only exists for square, non-singular matrices

### Eigenvalues and Eigenvectors: Principal Directions

For a square matrix A, eigenvector v and eigenvalue λ satisfy:
```
Av = λv
```

**Intuition**: 
- Eigenvectors are special directions that A only stretches (doesn't rotate)
- Eigenvalues tell us how much stretching occurs
- Principal Component Analysis finds directions of maximum variance

**Geometric Interpretation**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Example: 2x2 matrix with clear eigenvectors
A = np.array([[3, 1], [0, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)  # [3, 2]
print("Eigenvectors:\n", eigenvectors)
```

### Singular Value Decomposition (SVD): The Ultimate Decomposition

Any matrix A can be decomposed as:
```
A = UΣVᵀ
```

Where:
- **U**: Left singular vectors (orthogonal)
- **Σ**: Diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: Right singular vectors (orthogonal)

**Applications**:
- **Dimensionality reduction**: Keep only largest singular values
- **Matrix approximation**: Low-rank approximations
- **Principal Component Analysis**: V contains principal components
- **Recommender systems**: Collaborative filtering

## 💻 Coding Projects

### Project 1: Linear Regression with Normal Equation

**Mathematical Foundation**:
For linear regression y = Xβ + ε, the optimal parameters are:
```
β = (XᵀX)⁻¹Xᵀy
```

**Implementation**: [See code/01_linear_regression.py](../code/01_linear_regression.py)

### Project 2: Principal Component Analysis (PCA)

**Mathematical Steps**:
1. Center the data: X_centered = X - mean(X)
2. Compute covariance matrix: C = (1/n)X_centered^T X_centered  
3. Find eigenvalues/eigenvectors of C
4. Sort by eigenvalue magnitude
5. Project data onto top k principal components

**Implementation**: [See code/01_pca.py](../code/01_pca.py)

## 🔬 Worked Examples

### Example 1: Understanding Matrix Multiplication as Transformation

Consider the transformation matrix:
```
A = [2  0]
    [0  1]
```

This matrix:
- Stretches the x-axis by factor 2
- Leaves the y-axis unchanged
- Transforms vector [1, 1] → [2, 1]

### Example 2: Eigenvalue Analysis

For matrix:
```
A = [4  1]
    [2  3]
```

**Finding eigenvalues**:
```
det(A - λI) = det([4-λ   1  ]) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = 0
                 [2   3-λ]
```

Solutions: λ₁ = 5, λ₂ = 2

**Physical interpretation**: Matrix A has two principal directions with stretching factors 5 and 2.

## 🧪 Practice Exercises

### Exercise 1: Vector Operations
Given vectors u = [3, 4] and v = [1, 2]:
1. Calculate ||u|| and ||v||
2. Find u · v
3. Calculate the angle between u and v
4. Find a unit vector in direction of u

### Exercise 2: Matrix Properties
For matrix A = [[2, 1], [3, 4]]:
1. Calculate A²
2. Find det(A)
3. Compute A⁻¹
4. Verify AA⁻¹ = I

### Exercise 3: Eigenanalysis
1. Find eigenvalues and eigenvectors of [[5, 3], [3, 5]]
2. Verify Av = λv for each eigenvalue/eigenvector pair
3. Geometrically interpret what this transformation does

**Solutions**: [See exercises/01_linear_algebra_solutions.md](../exercises/01_linear_algebra_solutions.md)

## 🔗 Connection to Machine Learning

### Neural Networks as Matrix Operations

A neural network layer transforms input x to output y via:
```
y = f(Wx + b)
```

Where:
- **W**: Weight matrix (learned transformation)
- **x**: Input vector (features)
- **b**: Bias vector (learned offset)
- **f**: Activation function (non-linearity)

### Why Linear Algebra Matters

1. **Computational Efficiency**: GPUs excel at matrix operations
2. **Geometric Intuition**: Understand what networks actually do
3. **Debugging**: Diagnose issues with weight matrices
4. **Architecture Design**: Design better network structures

## 📖 Recommended Resources

### Essential Reading
- **Mathematics for Machine Learning** (Deisenroth et al.) - Chapter 2 & 3
- **3Blue1Brown Essence of Linear Algebra** - Videos 1-7

### Deep Dives
- **Linear Algebra Done Right** (Axler) - For mathematical rigor
- **Matrix Cookbook** - Quick reference for matrix identities

## 🎯 Key Takeaways

1. **Vectors represent data points and directions in high-dimensional space**
2. **Matrices are transformations that can rotate, scale, and shear**
3. **Eigenvalues/eigenvectors reveal fundamental properties of transformations**
4. **SVD provides the ultimate decomposition for any matrix**
5. **Neural networks are compositions of linear transformations and non-linearities**

## ➡️ Next Steps

Ready for [Step 2: Calculus & Gradients](02_calculus_gradients.md)? You'll learn how neural networks actually learn through gradient-based optimization.

---

**🚀 Interactive Learning**: Try the [Linear Algebra Jupyter Notebook](../notebooks/01_linear_algebra.ipynb) for hands-on experimentation!