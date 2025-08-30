# Step 1: Linear Algebra - Exercise Solutions

## Exercise 1: Vector Operations

**Given vectors u = [3, 4] and v = [1, 2]:**

### 1. Calculate ||u|| and ||v||

**Solution:**
```python
import numpy as np

u = np.array([3, 4])
v = np.array([1, 2])

# L2 norm (magnitude)
norm_u = np.linalg.norm(u)  # or np.sqrt(3**2 + 4**2)
norm_v = np.linalg.norm(v)  # or np.sqrt(1**2 + 2**2)

print(f"||u|| = {norm_u}")  # 5.0
print(f"||v|| = {norm_v}")  # 2.236
```

**Manual calculation:**
- ||u|| = √(3² + 4²) = √(9 + 16) = √25 = 5
- ||v|| = √(1² + 2²) = √(1 + 4) = √5 ≈ 2.236

### 2. Find u · v

**Solution:**
```python
dot_product = np.dot(u, v)  # or u @ v
print(f"u · v = {dot_product}")  # 11
```

**Manual calculation:**
u · v = 3×1 + 4×2 = 3 + 8 = 11

### 3. Calculate the angle between u and v

**Solution:**
```python
# Using formula: cos(θ) = (u · v) / (||u|| ||v||)
cos_theta = dot_product / (norm_u * norm_v)
theta_radians = np.arccos(cos_theta)
theta_degrees = np.degrees(theta_radians)

print(f"cos(θ) = {cos_theta:.3f}")      # 0.982
print(f"θ = {theta_degrees:.1f}°")      # 11.3°
```

**Manual calculation:**
- cos(θ) = 11 / (5 × √5) = 11 / (5√5) = 11√5 / 25 ≈ 0.982
- θ = arccos(0.982) ≈ 11.3°

### 4. Find a unit vector in direction of u

**Solution:**
```python
unit_u = u / norm_u
print(f"Unit vector in direction of u: {unit_u}")  # [0.6, 0.8]

# Verify it's a unit vector
print(f"Magnitude of unit vector: {np.linalg.norm(unit_u)}")  # 1.0
```

**Manual calculation:**
- Unit vector = u / ||u|| = [3, 4] / 5 = [0.6, 0.8]

---

## Exercise 2: Matrix Properties

**For matrix A = [[2, 1], [3, 4]]:**

### 1. Calculate A²

**Solution:**
```python
A = np.array([[2, 1], [3, 4]])
A_squared = A @ A  # or np.matmul(A, A)

print("A² =")
print(A_squared)
# [[7, 6],
#  [18, 19]]
```

**Manual calculation:**
```
A² = [[2, 1], [3, 4]] @ [[2, 1], [3, 4]]
   = [[2×2+1×3, 2×1+1×4], [3×2+4×3, 3×1+4×4]]
   = [[4+3, 2+4], [6+12, 3+16]]
   = [[7, 6], [18, 19]]
```

### 2. Find det(A)

**Solution:**
```python
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")  # 5.0
```

**Manual calculation:**
det(A) = 2×4 - 1×3 = 8 - 3 = 5

### 3. Compute A⁻¹

**Solution:**
```python
A_inv = np.linalg.inv(A)
print("A⁻¹ =")
print(A_inv)
# [[ 0.8, -0.2],
#  [-0.6,  0.4]]
```

**Manual calculation:**
For 2×2 matrix [[a, b], [c, d]], inverse is:
```
A⁻¹ = (1/det(A)) × [[d, -b], [-c, a]]
    = (1/5) × [[4, -1], [-3, 2]]
    = [[4/5, -1/5], [-3/5, 2/5]]
    = [[0.8, -0.2], [-0.6, 0.4]]
```

### 4. Verify AA⁻¹ = I

**Solution:**
```python
identity_check = A @ A_inv
print("AA⁻¹ =")
print(identity_check)
# [[1., 0.],
#  [0., 1.]]

# Check if it's close to identity (accounting for floating point errors)
is_identity = np.allclose(identity_check, np.eye(2))
print(f"Is AA⁻¹ = I? {is_identity}")  # True
```

---

## Exercise 3: Eigenanalysis

**For matrix [[5, 3], [3, 5]]:**

### 1. Find eigenvalues and eigenvectors

**Solution:**
```python
B = np.array([[5, 3], [3, 5]])
eigenvalues, eigenvectors = np.linalg.eig(B)

print("Eigenvalues:", eigenvalues)    # [8., 2.]
print("Eigenvectors:")
print(eigenvectors)
# [[ 0.707,  0.707],
#  [ 0.707, -0.707]]
```

**Manual calculation:**

**Step 1: Find eigenvalues**
Solve det(B - λI) = 0:
```
det([[5-λ, 3], [3, 5-λ]]) = (5-λ)² - 9 = 0
λ² - 10λ + 25 - 9 = 0
λ² - 10λ + 16 = 0
(λ - 8)(λ - 2) = 0
```
So λ₁ = 8, λ₂ = 2

**Step 2: Find eigenvectors**

For λ₁ = 8:
```
(B - 8I)v = 0
[[-3, 3], [3, -3]]v = 0
```
This gives us v₁ = [1, 1] (normalized: [√2/2, √2/2])

For λ₂ = 2:
```
(B - 2I)v = 0
[[3, 3], [3, 3]]v = 0
```
This gives us v₂ = [1, -1] (normalized: [√2/2, -√2/2])

### 2. Verify Av = λv for each eigenvalue/eigenvector pair

**Solution:**
```python
print("Verification:")
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    
    Bv = B @ v_i
    lambda_v = lambda_i * v_i
    
    print(f"\nFor λ = {lambda_i:.1f}:")
    print(f"Bv = {Bv}")
    print(f"λv = {lambda_v}")
    print(f"Equal? {np.allclose(Bv, lambda_v)}")
```

### 3. Geometrically interpret what this transformation does

**Solution:**

The matrix B = [[5, 3], [3, 5]] represents a transformation that:

1. **Stretches along the first eigenvector [1, 1] by factor 8**
   - This is the diagonal direction (45° line)
   - Points along this direction get stretched most

2. **Stretches along the second eigenvector [1, -1] by factor 2**
   - This is the anti-diagonal direction
   - Points along this direction get stretched less

3. **Overall effect:**
   - The transformation is symmetric (B = Bᵀ)
   - It preserves angles between eigenvectors (they remain orthogonal)
   - It's a positive definite transformation (all eigenvalues > 0)

**Visualization code:**
```python
import matplotlib.pyplot as plt

# Original unit circle
theta = np.linspace(0, 2*np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)])

# Transform the unit circle
transformed_circle = B @ unit_circle

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(unit_circle[0], unit_circle[1], 'b-', label='Original')
plt.plot(transformed_circle[0], transformed_circle[1], 'r-', label='Transformed')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Unit Circle Transformation')

plt.subplot(1, 2, 2)
# Plot eigenvectors
origin = np.array([[0, 0], [0, 0]])
plt.quiver(*origin, eigenvectors[0], eigenvectors[1], 
           scale=1, scale_units='xy', angles='xy', 
           color=['red', 'blue'], width=0.005)
plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='Eigenvector 1 (λ=8)')
plt.plot([-1, 1], [1, -1], 'b--', alpha=0.5, label='Eigenvector 2 (λ=2)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Principal Directions')

plt.tight_layout()
plt.show()
```

**Key insights:**
- The transformation creates an ellipse from a circle
- The major axis of the ellipse aligns with the first eigenvector
- The ratio of the axes is λ₁/λ₂ = 8/2 = 4
- This is exactly what PCA captures: the directions of maximum variance!

---

## Additional Practice Problems

### Problem 4: Matrix Rank and Linear Independence

**Given matrix C = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]:**

Find the rank and determine which rows/columns are linearly independent.

**Solution:**
```python
C = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
rank_C = np.linalg.matrix_rank(C)
print(f"Rank of C: {rank_C}")  # 2

# The second row is 2 times the first row, so they're linearly dependent
# Only the first and third rows are linearly independent
```

### Problem 5: Projection

**Project vector v = [3, 4] onto vector u = [1, 0]:**

**Solution:**
```python
v = np.array([3, 4])
u = np.array([1, 0])

# Projection formula: proj_u(v) = ((v·u) / (u·u)) * u
projection = ((v @ u) / (u @ u)) * u
print(f"Projection of v onto u: {projection}")  # [3, 0]

# This makes sense: projecting onto x-axis just keeps x-component
```

### Problem 6: Orthogonalization

**Use Gram-Schmidt to orthogonalize vectors [1, 1, 0] and [1, 0, 1]:**

**Solution:**
```python
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])

# Gram-Schmidt process
u1 = v1 / np.linalg.norm(v1)  # Normalize first vector

# Remove component of v2 in direction of u1
projection = (v2 @ u1) * u1
v2_orthogonal = v2 - projection
u2 = v2_orthogonal / np.linalg.norm(v2_orthogonal)

print(f"Orthonormal basis:")
print(f"u1 = {u1}")  # [0.707, 0.707, 0]
print(f"u2 = {u2}")  # [0.5, -0.5, 0.707]

# Verify orthogonality
print(f"u1 · u2 = {u1 @ u2}")  # ~0 (within numerical precision)
```

---

## Key Learning Points

1. **Vector operations** form the foundation of all linear algebra computations
2. **Matrix multiplication** represents composition of linear transformations  
3. **Eigenvalues and eigenvectors** reveal the fundamental behavior of matrices
4. **Geometric intuition** helps understand algebraic operations
5. **Numerical precision** matters in real implementations

These exercises demonstrate the core concepts that make machine learning possible!