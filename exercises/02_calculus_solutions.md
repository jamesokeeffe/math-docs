# Step 2: Calculus & Gradients - Exercise Solutions

## Exercise 1: Derivative Computation

### 1. f(x) = 3x⁴ - 2x³ + x² - 5x + 1

**Solution using power rule:**
- d/dx[3x⁴] = 3 · 4x³ = 12x³
- d/dx[-2x³] = -2 · 3x² = -6x²
- d/dx[x²] = 2x
- d/dx[-5x] = -5
- d/dx[1] = 0

**Final answer:** f'(x) = 12x³ - 6x² + 2x - 5

**Verification:**
```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**4 - 2*x**3 + x**2 - 5*x + 1

def f_prime(x):
    return 12*x**3 - 6*x**2 + 2*x - 5

# Numerical verification using finite differences
def numerical_derivative(f, x, h=1e-8):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test at several points
test_points = [-2, -1, 0, 1, 2]
for x in test_points:
    analytical = f_prime(x)
    numerical = numerical_derivative(f, x)
    print(f"x = {x}: Analytical = {analytical:.6f}, Numerical = {numerical:.6f}")
```

### 2. g(x) = e^x sin(x)

**Solution using product rule:** (uv)' = u'v + uv'
- u = e^x, u' = e^x
- v = sin(x), v' = cos(x)

**g'(x) = e^x · sin(x) + e^x · cos(x) = e^x(sin(x) + cos(x))**

**Verification:**
```python
import numpy as np

def g(x):
    return np.exp(x) * np.sin(x)

def g_prime(x):
    return np.exp(x) * (np.sin(x) + np.cos(x))

# Test at x = π/4 where sin(x) = cos(x) = √2/2
x_test = np.pi / 4
analytical = g_prime(x_test)
numerical = numerical_derivative(g, x_test)
print(f"At x = π/4: Analytical = {analytical:.6f}, Numerical = {numerical:.6f}")
print(f"Should equal: e^(π/4) * √2 = {np.exp(np.pi/4) * np.sqrt(2):.6f}")
```

### 3. h(x) = ln(x² + 1)

**Solution using chain rule:** If h(x) = f(g(x)), then h'(x) = f'(g(x)) · g'(x)
- Let u = x² + 1, so h(x) = ln(u)
- dh/du = 1/u
- du/dx = 2x
- By chain rule: dh/dx = (dh/du)(du/dx) = (1/u)(2x) = 2x/(x² + 1)

**h'(x) = 2x/(x² + 1)**

**Verification:**
```python
def h(x):
    return np.log(x**2 + 1)

def h_prime(x):
    return 2*x / (x**2 + 1)

# Test at several points
test_points = [-2, -1, 0, 1, 2]
for x in test_points:
    analytical = h_prime(x)
    numerical = numerical_derivative(h, x)
    print(f"x = {x}: Analytical = {analytical:.6f}, Numerical = {numerical:.6f}")
```

---

## Exercise 2: Gradient Computation

**For f(x,y) = x³y² + 2xy - y³:**

### 1. Find ∂f/∂x and ∂f/∂y

**∂f/∂x** (treat y as constant):
- ∂/∂x[x³y²] = 3x²y²
- ∂/∂x[2xy] = 2y
- ∂/∂x[-y³] = 0

**∂f/∂x = 3x²y² + 2y**

**∂f/∂y** (treat x as constant):
- ∂/∂y[x³y²] = x³ · 2y = 2x³y
- ∂/∂y[2xy] = 2x
- ∂/∂y[-y³] = -3y²

**∂f/∂y = 2x³y + 2x - 3y²**

### 2. Compute ∇f at point (1, -1)

**∇f = [∂f/∂x, ∂f/∂y] = [3x²y² + 2y, 2x³y + 2x - 3y²]**

At (1, -1):
- ∂f/∂x = 3(1)²(-1)² + 2(-1) = 3(1)(1) - 2 = 3 - 2 = 1
- ∂f/∂y = 2(1)³(-1) + 2(1) - 3(-1)² = -2 + 2 - 3 = -3

**∇f(1, -1) = [1, -3]**

### 3. Find critical points where ∇f = 0

Set both partial derivatives to zero:
- 3x²y² + 2y = 0  →  y(3x²y + 2) = 0
- 2x³y + 2x - 3y² = 0

From the first equation: either y = 0 or y = -2/(3x²)

**Case 1: y = 0**
Substitute into second equation:
2x³(0) + 2x - 3(0)² = 0
2x = 0
x = 0

So (0, 0) is a critical point.

**Case 2: y = -2/(3x²)**
Substitute into second equation:
2x³(-2/(3x²)) + 2x - 3(-2/(3x²))² = 0
-4x/3 + 2x - 3(4/(9x⁴)) = 0
-4x/3 + 2x - 4/(3x⁴) = 0

Multiply by 3x⁴:
-4x⁵ + 6x⁵ - 4 = 0
2x⁵ = 4
x⁵ = 2
x = 2^(1/5) ≈ 1.149

Then y = -2/(3(2^(1/5))²) = -2/(3·2^(2/5)) ≈ -0.504

**Critical points: (0, 0) and (2^(1/5), -2/(3·2^(2/5)))**

**Verification:**
```python
def f(x, y):
    return x**3 * y**2 + 2*x*y - y**3

def gradient_f(x, y):
    df_dx = 3*x**2 * y**2 + 2*y
    df_dy = 2*x**3 * y + 2*x - 3*y**2
    return np.array([df_dx, df_dy])

# Verify critical points
critical_points = [(0, 0), (2**(1/5), -2/(3*2**(2/5)))]

for point in critical_points:
    grad = gradient_f(point[0], point[1])
    print(f"∇f{point} = [{grad[0]:.6f}, {grad[1]:.6f}]")
```

---

## Exercise 3: Chain Rule Practice

**For z = e^(x²+y²) where x = t² and y = t³:**

### 1. Find dz/dt using the chain rule

**Method 1: Direct substitution then differentiate**
z = e^(x²+y²) = e^((t²)²+(t³)²) = e^(t⁴+t⁶)

dz/dt = e^(t⁴+t⁶) · d/dt[t⁴ + t⁶] = e^(t⁴+t⁶) · (4t³ + 6t⁵)

**Method 2: Multivariable chain rule**
dz/dt = ∂z/∂x · dx/dt + ∂z/∂y · dy/dt

First, find the partial derivatives:
- ∂z/∂x = e^(x²+y²) · 2x
- ∂z/∂y = e^(x²+y²) · 2y
- dx/dt = 2t
- dy/dt = 3t²

Now apply chain rule:
dz/dt = e^(x²+y²) · 2x · 2t + e^(x²+y²) · 2y · 3t²
     = e^(x²+y²) · (4xt + 6yt²)

Substitute x = t² and y = t³:
dz/dt = e^(t⁴+t⁶) · (4t² · t + 6t³ · t²)
     = e^(t⁴+t⁶) · (4t³ + 6t⁵)

Both methods give the same result! ✓

### 2. Evaluate at t = 1

At t = 1:
- x = 1² = 1
- y = 1³ = 1
- z = e^(1²+1²) = e²

dz/dt|_{t=1} = e^(1⁴+1⁶) · (4(1)³ + 6(1)⁵) = e² · (4 + 6) = 10e²

**Numerical value:** 10e² ≈ 10 × 7.389 ≈ 73.89

**Verification:**
```python
def z_of_t(t):
    x = t**2
    y = t**3
    return np.exp(x**2 + y**2)

def dz_dt_analytical(t):
    return np.exp(t**4 + t**6) * (4*t**3 + 6*t**5)

# Evaluate at t = 1
t = 1
analytical = dz_dt_analytical(t)
numerical = numerical_derivative(z_of_t, t)

print(f"At t = 1:")
print(f"Analytical dz/dt = {analytical:.6f}")
print(f"Numerical dz/dt = {numerical:.6f}")
print(f"Expected: 10e² = {10 * np.exp(2):.6f}")
```

---

## Additional Practice Problems

### Problem 4: Backpropagation Example

**Simple network:** x → w₁ → σ(·) → w₂ → y
**Loss:** L = ½(y - target)²

Given: x = 2, w₁ = 0.5, w₂ = -1.5, target = 1
Activation function: σ(z) = 1/(1 + e^(-z))

**Forward pass:**
1. z₁ = w₁x = 0.5 × 2 = 1
2. a₁ = σ(z₁) = σ(1) = 1/(1 + e^(-1)) ≈ 0.731
3. y = w₂a₁ = -1.5 × 0.731 ≈ -1.097
4. L = ½(y - target)² = ½(-1.097 - 1)² = ½(-2.097)² ≈ 2.199

**Backward pass:**
1. ∂L/∂y = y - target = -1.097 - 1 = -2.097
2. ∂L/∂w₂ = ∂L/∂y × ∂y/∂w₂ = -2.097 × a₁ = -2.097 × 0.731 ≈ -1.533
3. ∂L/∂a₁ = ∂L/∂y × ∂y/∂a₁ = -2.097 × w₂ = -2.097 × (-1.5) = 3.146
4. ∂L/∂z₁ = ∂L/∂a₁ × ∂a₁/∂z₁ = 3.146 × σ'(z₁)

For sigmoid: σ'(z) = σ(z)(1 - σ(z))
σ'(1) = 0.731 × (1 - 0.731) = 0.731 × 0.269 ≈ 0.197

5. ∂L/∂z₁ = 3.146 × 0.197 ≈ 0.620
6. ∂L/∂w₁ = ∂L/∂z₁ × ∂z₁/∂w₁ = 0.620 × x = 0.620 × 2 = 1.240

**Final gradients:**
- ∂L/∂w₁ ≈ 1.240
- ∂L/∂w₂ ≈ -1.533

### Problem 5: Optimization Landscape Analysis

**Function:** f(x,y) = x² + 10y²

**Analysis:**
1. **Gradient:** ∇f = [2x, 20y]
2. **Critical point:** (0, 0) where ∇f = 0
3. **Hessian matrix:** H = [[2, 0], [0, 20]]
4. **Eigenvalues:** λ₁ = 2, λ₂ = 20
5. **Condition number:** κ = λ₂/λ₁ = 10

This function has an **ill-conditioned** optimization landscape:
- Steep in y-direction (eigenvalue 20)
- Gentle in x-direction (eigenvalue 2)
- Causes slow convergence with gradient descent

**Gradient descent behavior:**
- Large steps overshoot in x-direction
- Small steps under-progress in y-direction
- Results in zigzag pattern toward optimum

```python
def ill_conditioned(x, y):
    return x**2 + 10*y**2

def ill_conditioned_grad(point):
    x, y = point
    return np.array([2*x, 20*y])

# Visualize optimization path
from gradient_descent import GradientDescent

optimizer = GradientDescent(learning_rate=0.1, max_iterations=100)
x_opt, f_opt = optimizer.optimize(
    lambda p: ill_conditioned(p[0], p[1]),
    ill_conditioned_grad,
    np.array([5.0, 1.0])
)

# Plot the zigzag convergence pattern
```

---

## Key Learning Points

1. **Power rule** handles polynomial terms efficiently
2. **Product rule** for products of functions: (uv)' = u'v + uv'
3. **Chain rule** enables differentiation of composite functions
4. **Partial derivatives** extend calculus to multivariable functions
5. **Gradient vectors** point toward steepest ascent
6. **Critical points** occur where gradient equals zero
7. **Chain rule in ML** enables efficient backpropagation
8. **Condition numbers** affect optimization difficulty

These concepts form the mathematical foundation that makes neural network training possible!