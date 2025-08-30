# Step 3: Probability & Statistics – Data & Uncertainty

## Overview

Probability and statistics form the foundation for handling uncertainty in machine learning. Every prediction comes with uncertainty, every loss function has probabilistic interpretation, and every model makes assumptions about data distributions. In this step, you'll master the probabilistic thinking essential for modern AI.

## 🎯 Learning Objectives

By the end of this step, you will:
- Master random variables, probability distributions, and their properties
- Understand expectation, variance, and higher-order moments
- Work confidently with common distributions (Bernoulli, Normal, Categorical)
- Apply Bayes' theorem for probabilistic reasoning
- Implement and understand cross-entropy loss from first principles
- Build probabilistic classifiers (Naive Bayes, Logistic Regression)
- Connect statistical concepts to machine learning algorithms

## 📚 Mathematical Foundations

### Random Variables and Probability

A **random variable** X is a function that assigns numerical values to outcomes of random experiments.

**Discrete Random Variables**:
- Probability Mass Function (PMF): P(X = x)
- Properties: P(X = x) ≥ 0, Σₓ P(X = x) = 1

**Continuous Random Variables**:
- Probability Density Function (PDF): f(x)
- Properties: f(x) ≥ 0, ∫ f(x)dx = 1
- P(a ≤ X ≤ b) = ∫ᵃᵇ f(x)dx

### Expectation and Variance

**Expectation (Mean)**:
```
E[X] = Σₓ x·P(X = x)     (discrete)
E[X] = ∫ x·f(x)dx        (continuous)
```

**Variance**:
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Standard Deviation**: σ = √Var(X)

**Key Properties**:
- E[aX + b] = aE[X] + b
- Var(aX + b) = a²Var(X)
- For independent X, Y: E[XY] = E[X]E[Y]

### Common Probability Distributions

#### Bernoulli Distribution
Models a single binary trial (coin flip):
```
X ~ Bernoulli(p)
P(X = 1) = p, P(X = 0) = 1-p
E[X] = p, Var(X) = p(1-p)
```

#### Binomial Distribution
Number of successes in n independent Bernoulli trials:
```
X ~ Binomial(n, p)
P(X = k) = C(n,k) p^k (1-p)^(n-k)
E[X] = np, Var(X) = np(1-p)
```

#### Normal (Gaussian) Distribution
The most important continuous distribution:
```
X ~ Normal(μ, σ²)
f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
E[X] = μ, Var(X) = σ²
```

#### Categorical Distribution
Generalizes Bernoulli to k categories:
```
X ~ Categorical(p₁, p₂, ..., pₖ)
P(X = i) = pᵢ, where Σᵢ pᵢ = 1
```

### Conditional Probability and Bayes' Theorem

**Conditional Probability**:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem**:
```
P(A|B) = P(B|A)P(A) / P(B)
```

**In machine learning context**:
```
P(class|features) = P(features|class)P(class) / P(features)
```

### Information Theory Fundamentals

#### Entropy
Measures uncertainty in a random variable:
```
H(X) = -Σₓ P(X = x) log P(X = x)
```

**Properties**:
- H(X) ≥ 0
- H(X) is maximized when X is uniformly distributed
- H(X) = 0 when X is deterministic

#### Cross-Entropy
Measures the difference between two probability distributions:
```
H(p, q) = -Σₓ p(x) log q(x)
```

**In classification**: Cross-entropy loss measures how far predicted probabilities are from true labels.

#### KL Divergence
Measures how one distribution differs from another:
```
D_KL(P||Q) = Σₓ P(x) log(P(x)/Q(x))
```

**Properties**:
- D_KL(P||Q) ≥ 0
- D_KL(P||Q) = 0 iff P = Q
- Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

## 💻 Coding Projects

### Project 1: Logistic Regression from Scratch

**Mathematical Foundation**:
Logistic regression models P(y=1|x) using the sigmoid function:
```
P(y=1|x) = σ(wᵀx + b) = 1/(1 + e^(-(wᵀx + b)))
```

**Loss Function**: Negative log-likelihood (cross-entropy)
```
L = -Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Implementation**: [See code/03_logistic_regression.py](../code/03_logistic_regression.py)

### Project 2: Naive Bayes Text Classifier

**Mathematical Foundation**:
Assumes features are conditionally independent given the class:
```
P(class|features) ∝ P(class) ∏ᵢ P(featureᵢ|class)
```

**For text classification**:
```
P(class|words) ∝ P(class) ∏ᵢ P(wordᵢ|class)
```

**Implementation**: [See code/03_naive_bayes.py](../code/03_naive_bayes.py)

### Project 3: Custom Cross-Entropy Loss

**Mathematical Steps**:
1. Implement softmax function for multi-class probabilities
2. Compute cross-entropy loss
3. Derive and implement gradients
4. Compare with standard implementations

**Implementation**: [See code/03_cross_entropy.py](../code/03_cross_entropy.py)

## 🔬 Worked Examples

### Example 1: Bayes' Theorem in Medical Diagnosis

**Problem**: A medical test for a rare disease:
- Disease affects 1% of population: P(Disease) = 0.01
- Test accuracy: P(Positive|Disease) = 0.95
- False positive rate: P(Positive|No Disease) = 0.02

**Question**: If test is positive, what's P(Disease|Positive)?

**Solution using Bayes' theorem**:
```
P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)

P(Positive) = P(Positive|Disease) × P(Disease) + P(Positive|No Disease) × P(No Disease)
            = 0.95 × 0.01 + 0.02 × 0.99
            = 0.0095 + 0.0198 = 0.0293

P(Disease|Positive) = (0.95 × 0.01) / 0.0293 = 0.324
```

**Interpretation**: Even with a positive test, there's only 32.4% chance of having the disease!

### Example 2: Maximum Likelihood Estimation

**Problem**: Given data points x₁, x₂, ..., xₙ from Normal(μ, σ²), find μ̂ and σ̂².

**Likelihood function**:
```
L(μ, σ²) = ∏ᵢ (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))
```

**Log-likelihood**:
```
ℓ(μ, σ²) = -n/2 log(2π) - n/2 log(σ²) - Σᵢ(xᵢ-μ)²/(2σ²)
```

**MLE solutions**:
```
μ̂ = (1/n) Σᵢ xᵢ  (sample mean)
σ̂² = (1/n) Σᵢ(xᵢ-μ̂)²  (sample variance)
```

### Example 3: Cross-Entropy Gradient Derivation

**For binary classification with sigmoid output**:
```
ŷ = σ(z) = 1/(1 + e^(-z))
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Gradient computation**:
```
∂L/∂z = ∂L/∂ŷ × ∂ŷ/∂z

∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
∂ŷ/∂z = σ(z)(1-σ(z)) = ŷ(1-ŷ)

∂L/∂z = [-y/ŷ + (1-y)/(1-ŷ)] × ŷ(1-ŷ)
       = -y(1-ŷ) + (1-y)ŷ
       = -y + yŷ + ŷ - yŷ
       = ŷ - y
```

**Beautiful result**: The gradient is simply (prediction - target)!

## 🧪 Practice Exercises

### Exercise 1: Distribution Properties
For X ~ Normal(3, 4):
1. Find P(X ≤ 5)
2. Find P(1 ≤ X ≤ 5)
3. What is the 95th percentile?

### Exercise 2: Bayes' Theorem
A spam filter has:
- P(spam) = 0.4
- P(word "free" | spam) = 0.8
- P(word "free" | not spam) = 0.1

If an email contains "free", what's P(spam | "free")?

### Exercise 3: Entropy Calculation
For a distribution P = [0.5, 0.3, 0.2]:
1. Calculate entropy H(P)
2. Calculate cross-entropy H(P, Q) where Q = [0.4, 0.4, 0.2]
3. Calculate KL divergence D_KL(P||Q)

**Solutions**: [See exercises/03_probability_solutions.md](../exercises/03_probability_solutions.md)

## 🔗 Connection to Machine Learning

### Loss Functions as Negative Log-Likelihood

**Mean Squared Error** corresponds to Gaussian assumption:
```
P(y|x) ~ Normal(f(x), σ²)
L = -log P(y|x) ∝ (y - f(x))²
```

**Cross-Entropy Loss** corresponds to Bernoulli/Categorical assumption:
```
P(y|x) ~ Bernoulli(σ(f(x)))
L = -log P(y|x) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

### Regularization as Prior Beliefs

**L2 regularization** corresponds to Gaussian prior on weights:
```
P(w) ~ Normal(0, λ⁻¹I)
L = -log P(data|w) - log P(w) ∝ MSE + λ||w||²
```

**L1 regularization** corresponds to Laplace prior:
```
P(w) ~ Laplace(0, λ⁻¹)
L = -log P(data|w) - log P(w) ∝ MSE + λ||w||₁
```

### Uncertainty Quantification

**Epistemic Uncertainty**: Model uncertainty (reducible with more data)
**Aleatoric Uncertainty**: Data noise (irreducible)

**Bayesian Neural Networks** model weight distributions to capture epistemic uncertainty.

## 📖 Recommended Resources

### Essential Reading
- **Mathematics for Machine Learning** (Deisenroth et al.) - Chapter 6
- **Deep Learning** (Goodfellow et al.) - Chapter 3
- **Pattern Recognition and Machine Learning** (Bishop) - Chapters 1-2

### Deep Dives
- **Information Theory, Inference, and Learning Algorithms** (MacKay)
- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)

## 🎯 Key Takeaways

1. **Probability quantifies uncertainty** - essential for realistic predictions
2. **Bayes' theorem enables probabilistic reasoning** - foundation of probabilistic ML
3. **Maximum likelihood connects statistics to optimization** - basis of most loss functions
4. **Cross-entropy measures distributional distance** - why it's the standard classification loss
5. **Information theory provides principled measures** - entropy, KL divergence guide model design
6. **Probabilistic interpretation explains regularization** - priors become regularizers

## ➡️ Next Steps

Ready for [Step 4: Optimization](04_optimization.md)? You'll learn how to make training efficient and robust with advanced optimization techniques.

---

**🚀 Interactive Learning**: Try the [Probability & Statistics Jupyter Notebook](../notebooks/03_probability_statistics.ipynb) for hands-on exploration of distributions and probabilistic reasoning!