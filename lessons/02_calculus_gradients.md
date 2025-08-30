# Step 2: Calculus & Gradients – How Models Learn

## Overview

Calculus is the mathematical engine of machine learning. Every time a neural network adjusts its weights to improve performance, it's using calculus—specifically, gradients and the chain rule. In this step, you'll understand the mathematical principles behind backpropagation and gradient descent.

## 🎯 Learning Objectives

By the end of this step, you will:
- Master derivatives and partial derivatives conceptually and computationally
- Understand the chain rule as the foundation of backpropagation
- Compute gradients, Jacobians, and Hessians for optimization
- Implement gradient descent from scratch and visualize convergence
- Build a 2-layer neural network with manual forward and backward passes
- Train a neural network on MNIST using your own implementation

## 📚 Mathematical Foundations

### Derivatives: Measuring Change

A derivative measures how a function changes as its input changes:
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Geometric Interpretation**: The slope of the tangent line at point x.

**Key Properties**:
- **Power rule**: d/dx[x^n] = nx^(n-1)
- **Product rule**: d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
- **Chain rule**: d/dx[f(g(x))] = f'(g(x)) · g'(x)

### Partial Derivatives: Multivariable Functions

For functions with multiple variables, partial derivatives measure change with respect to one variable while holding others constant:

```
∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h
```

**Example**: For f(x,y) = x²y + 3xy²
- ∂f/∂x = 2xy + 3y²
- ∂f/∂y = x² + 6xy

### The Gradient: Direction of Steepest Ascent

The gradient combines all partial derivatives into a vector:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Key Properties**:
- **Direction**: Points toward steepest increase
- **Magnitude**: Rate of increase in that direction
- **Negative gradient**: Points toward steepest decrease (used in optimization)

### Chain Rule: The Heart of Backpropagation

For composite functions z = f(g(x)), the chain rule states:
```
dz/dx = dz/dg · dg/dx
```

**Multivariable chain rule** (crucial for neural networks):
If z = f(u, v) where u = g(x, y) and v = h(x, y), then:
```
∂z/∂x = ∂z/∂u · ∂u/∂x + ∂z/∂v · ∂v/∂x
```

### Gradient Descent: Following the Slope Downhill

Gradient descent is an optimization algorithm that finds function minima:
```
x_{t+1} = x_t - α∇f(x_t)
```

Where:
- **α**: Learning rate (step size)
- **∇f(x_t)**: Gradient at current point
- **Intuition**: Move in the direction opposite to the gradient

## 🧠 Neural Network Mathematics

### Forward Pass: Computing Outputs

For a 2-layer network:
```
Layer 1: z₁ = W₁x + b₁
         a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂
         a₂ = σ(z₂)
```

### Loss Function: Measuring Error

Mean Squared Error for regression:
```
L = (1/2n) Σᵢ (yᵢ - ŷᵢ)²
```

Cross-entropy for classification:
```
L = -Σᵢ yᵢ log(ŷᵢ)
```

### Backward Pass: Computing Gradients

Using the chain rule to compute gradients:

```
∂L/∂W₂ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂
∂L/∂W₁ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

This is **backpropagation**: efficiently computing all gradients in one backward pass.

## 💻 Coding Projects

### Project 1: Gradient Descent on Quadratic Function

**Mathematical Foundation**:
For f(x) = ax² + bx + c, the gradient is f'(x) = 2ax + b.
Gradient descent: x_{t+1} = x_t - α(2ax_t + b)

**Implementation**: [See code/02_gradient_descent.py](../code/02_gradient_descent.py)

### Project 2: 2-Layer Neural Network from Scratch

**Mathematical Steps**:
1. **Forward pass**: Compute activations layer by layer
2. **Loss computation**: Calculate error using loss function
3. **Backward pass**: Use chain rule to compute gradients
4. **Weight update**: Apply gradient descent to weights

**Implementation**: [See code/02_neural_network.py](../code/02_neural_network.py)

### Project 3: MNIST Training

**Application**: Train the neural network on handwritten digit recognition.

**Implementation**: [See code/02_mnist_training.py](../code/02_mnist_training.py)

## 🔬 Worked Examples

### Example 1: Computing Gradients by Hand

**Function**: f(x,y) = x² + 2xy + y²

**Gradients**:
- ∂f/∂x = 2x + 2y
- ∂f/∂y = 2x + 2y
- ∇f = [2x + 2y, 2x + 2y]

**At point (1,2)**:
- ∇f(1,2) = [2(1) + 2(2), 2(1) + 2(2)] = [6, 6]

### Example 2: Chain Rule in Action

**Composite function**: z = sin(x² + y²)

**Step-by-step**:
1. Let u = x² + y²
2. Then z = sin(u)
3. ∂z/∂x = ∂z/∂u · ∂u/∂x = cos(u) · 2x = 2x·cos(x² + y²)
4. ∂z/∂y = ∂z/∂u · ∂u/∂y = cos(u) · 2y = 2y·cos(x² + y²)

### Example 3: Backpropagation Through Simple Network

**Network**: x → W₁ → σ → W₂ → y
**Loss**: L = ½(y - target)²

**Forward**:
- z₁ = W₁x
- a₁ = σ(z₁)
- y = W₂a₁

**Backward**:
- ∂L/∂y = y - target
- ∂L/∂W₂ = ∂L/∂y · a₁ = (y - target) · a₁
- ∂L/∂a₁ = ∂L/∂y · W₂ = (y - target) · W₂
- ∂L/∂W₁ = ∂L/∂a₁ · σ'(z₁) · x = (y - target) · W₂ · σ'(z₁) · x

## 🧪 Practice Exercises

### Exercise 1: Derivative Computation
Compute derivatives for:
1. f(x) = 3x⁴ - 2x³ + x² - 5x + 1
2. g(x) = e^x sin(x)
3. h(x) = ln(x² + 1)

### Exercise 2: Gradient Computation
For f(x,y) = x³y² + 2xy - y³:
1. Find ∂f/∂x and ∂f/∂y
2. Compute ∇f at point (1, -1)
3. Find critical points where ∇f = 0

### Exercise 3: Chain Rule Practice
For z = e^(x²+y²) where x = t² and y = t³:
1. Find dz/dt using the chain rule
2. Evaluate at t = 1

**Solutions**: [See exercises/02_calculus_solutions.md](../exercises/02_calculus_solutions.md)

## 🔗 Connection to Machine Learning

### Why Gradients Matter

1. **Optimization**: Gradients point toward function minima/maxima
2. **Efficiency**: Backpropagation computes all gradients in O(n) time
3. **Generalization**: Same principles work for any differentiable function
4. **Scale**: Enables training networks with millions of parameters

### Common Activation Functions and Their Derivatives

```python
# Sigmoid
σ(x) = 1/(1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))

# ReLU
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0

# Tanh
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

### Gradient-Based Optimization Algorithms

1. **Vanilla Gradient Descent**: x ← x - α∇f(x)
2. **Momentum**: Accumulates gradients over time
3. **Adam**: Adaptive learning rates per parameter
4. **RMSprop**: Scales gradients by running average

## 📖 Recommended Resources

### Essential Reading
- **3Blue1Brown Essence of Calculus** - Intuitive visual explanations
- **Mathematics for Machine Learning** (Deisenroth et al.) - Chapter 4

### Deep Dives
- **Deep Learning** (Goodfellow et al.) - Chapter 6 on optimization
- **Pattern Recognition and Machine Learning** (Bishop) - Mathematical foundations

## 🎯 Key Takeaways

1. **Derivatives measure rates of change** - fundamental to optimization
2. **Chain rule enables efficient gradient computation** through complex networks
3. **Gradients point toward steepest ascent/descent** - basis of optimization
4. **Backpropagation is just chain rule applied systematically**
5. **Understanding calculus demystifies neural network training**

## ➡️ Next Steps

Ready for [Step 3: Probability & Statistics](03_probability_statistics.md)? You'll learn how uncertainty and randomness are modeled in machine learning.

---

**🚀 Interactive Learning**: Try the [Calculus & Gradients Jupyter Notebook](../notebooks/02_calculus_gradients.ipynb) for hands-on experimentation with gradient descent and backpropagation!