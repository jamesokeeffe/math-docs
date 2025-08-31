# AI Math → ML Implementation Guide
Educational repository providing hands-on learning of mathematical foundations of AI/ML through theory, code implementations, and interactive exercises.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Without Full Setup

### If Network Issues Prevent Package Installation
You can still work with the repository structure and documentation:

```bash
# Verify repository structure
ls -la  # Should show main directories and files

# Read documentation files
cat README.md
cat GETTING_STARTED.md  
cat IMPLEMENTATION_SUMMARY.md

# Examine code structure (without running)
ls code/
head -50 code/01_pca.py  # View working implementation

# Check lessons and exercises
ls lessons/ exercises/
cat lessons/01_linear_algebra.md
```

### Offline Validation Commands
```bash
# Verify Python is available
python3 --version  # Should show Python 3.x

# Check file structure matches expected layout
find . -path "./venv" -prune -o -path "./test_*" -prune -o -type f -name "*.py" -print | wc -l   # Should show 5 Python files
find . -path "./.git" -prune -o -type f -name "*.md" -print | wc -l   # Should show 6+ Markdown files  
find . -path "./venv" -prune -o -type f -name "*.ipynb" -print | wc -l # Should show 1 Jupyter notebook

# Verify key directories exist
test -d lessons && echo "lessons/ exists"
test -d code && echo "code/ exists"  
test -d exercises && echo "exercises/ exists"
test -d notebooks && echo "notebooks/ exists"
```

## Working Effectively

### Bootstrap and Setup (Required First Steps)
Run these commands to set up the development environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate    # On Windows

# Install dependencies - NEVER CANCEL: takes 45-180 seconds
pip install -r requirements.txt
```
**TIMEOUT SETTING**: Set timeout to 300+ seconds for pip install. NEVER CANCEL - installation takes 45-180 seconds to complete.

**NETWORK TIMEOUT WORKAROUND**: If pip install fails with network timeouts (common in CI environments):
```bash
# Use extended timeout and retries
pip install --timeout=300 --retries=5 -r requirements.txt

# Or install core packages first, then others
pip install numpy matplotlib scipy scikit-learn pandas
pip install jupyter torch torchvision plotly seaborn sympy

# Alternative: Use conda instead of pip in environments with network issues
conda install numpy matplotlib scipy scikit-learn pandas jupyter pytorch plotly seaborn sympy
```

**Note**: Network timeouts during pip install are common in CI/containerized environments. The above workarounds will help, but be prepared to wait longer or use alternative package managers.

### Validate Setup
Always validate your setup works before making changes:
```bash
# Activate environment first
source venv/bin/activate

# Test basic imports (should complete instantly)
python -c "import numpy, matplotlib, sklearn, torch, pandas; print('All imports successful!')"
```

### Run Working Code Examples
Test with known working implementations:
```bash
# Activate environment first
source venv/bin/activate

# PCA implementation - WORKS: takes ~4 seconds
time python code/01_pca.py

# Start Jupyter Lab - takes ~10 seconds, NEVER CANCEL
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

## Known Issues and Workarounds

### Broken Code Files (DO NOT RUN AS-IS)
These files have bugs and will fail:
- `code/01_linear_regression.py` - LinAlgError: Singular matrix
- `code/02_gradient_descent.py` - SyntaxError: unexpected character after line continuation  
- `notebooks/01_linear_algebra.ipynb` - ValueError: matmul dimension mismatch

When fixing these files:
- Always test after changes with `python [filename]`
- The linear regression matrix analysis demo has a singular matrix issue
- The gradient descent file has a line continuation character error around line 222
- The Jupyter notebook has grid transformation matrix dimension issues

### Missing Data Directory
The documentation references a `data/` directory that doesn't exist. If you need sample datasets:
- Use built-in datasets from sklearn: `from sklearn.datasets import load_iris, make_classification`
- Generate synthetic data with numpy for demonstrations

## Validation Scenarios

### Required Manual Testing After Changes
Always run these validation scenarios after making code changes:

1. **Environment Setup Test**:
   ```bash
   source venv/bin/activate
   python -c "import numpy, matplotlib, sklearn, torch, pandas; print('Setup successful!')"
   ```

2. **Working Code Test**:
   ```bash
   source venv/bin/activate
   python code/01_pca.py  # Should complete successfully with PCA output
   ```

3. **Repository Structure Test**:
   ```bash
   find . -path "./venv" -prune -o -path "./test_*" -prune -o -type f -name "*.py" -print | wc -l  # Should be 5
   test -d lessons && test -d code && test -d exercises && test -d notebooks && echo "All directories exist"
   ```

4. **Jupyter Environment Test**:
   ```bash
   source venv/bin/activate
   jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
   # Should start without errors and show server URL like:
   # http://127.0.0.1:8888/lab?token=... 
   # Press Ctrl+C to stop
   ```

5. **If You Fix Broken Code**:
   ```bash
   source venv/bin/activate
   python code/01_linear_regression.py  # Should run without LinAlgError
   python code/02_gradient_descent.py   # Should run without SyntaxError
   jupyter nbconvert --to python --execute notebooks/01_linear_algebra.ipynb  # Should execute without errors
   ```

### Validation Expectations
- **PCA code**: Should output explained variance ratios, transformation results, and mathematical insights
- **Jupyter Lab**: Should start in ~10 seconds and display a URL with authentication token
- **File counts**: 5 Python files in code/, 1 Jupyter notebook, 6+ Markdown files
- **Import test**: Should complete instantly without any import errors

## Repository Navigation

### Key Directories and Their Purpose
```
math-docs/
├── lessons/           # Mathematical theory and explanations
│   ├── 01_linear_algebra.md
│   ├── 02_calculus_gradients.md
│   └── 03_probability_statistics.md
├── code/             # Python implementations from scratch
│   ├── 01_linear_regression.py    # ❌ Has bugs (LinAlgError)
│   ├── 01_pca.py                  # ✅ Works correctly  
│   ├── 02_gradient_descent.py     # ❌ Has bugs (SyntaxError)
│   ├── 02_neural_network.py       # ⚠️  Untested
│   └── 03_logistic_regression.py  # ⚠️  Untested
├── exercises/        # Practice problems with solutions
│   ├── 01_linear_algebra_solutions.md
│   └── 02_calculus_solutions.md
├── notebooks/        # Interactive Jupyter notebooks
│   └── 01_linear_algebra.ipynb   # ❌ Has bugs (dimension errors)
├── requirements.txt  # Python dependencies
├── README.md         # Main overview and learning path
├── GETTING_STARTED.md # Detailed setup guide and learning strategy
├── IMPLEMENTATION_SUMMARY.md # Project status overview
└── .github/
    └── copilot-instructions.md   # These instructions
```

### Important Files
- **README.md**: Overview of the learning path and repository structure
- **GETTING_STARTED.md**: Comprehensive setup and learning guide with troubleshooting
- **requirements.txt**: All required Python packages (numpy, matplotlib, sklearn, torch, etc.)
- **IMPLEMENTATION_SUMMARY.md**: Current status of different components
- **task.md**: Original learning roadmap and project structure

## Learning Path Structure

The repository follows a 7-step learning path:
1. **Linear Algebra** - Vector operations, matrices, eigenvalues (lessons/01_linear_algebra.md)
2. **Calculus & Gradients** - Derivatives, backpropagation (lessons/02_calculus_gradients.md)  
3. **Probability & Statistics** - Distributions, uncertainty (lessons/03_probability_statistics.md)
4. **Optimization** - SGD, Adam, regularization (not yet implemented)
5. **Information Theory** - Entropy, attention mechanisms (not yet implemented)
6. **Numerical Methods** - Stability, precision (not yet implemented)
7. **Advanced Topics** - Specialization paths (not yet implemented)

## Common Tasks

### Adding New Code Examples
- Place in appropriate `code/` directory
- Follow naming convention: `[step_number]_[topic_name].py`
- Always test with: `python code/[filename].py`
- Include comprehensive docstrings and mathematical explanations

### Working with Notebooks
- Start Jupyter Lab: `jupyter lab --no-browser --port=8888`
- Access via provided URL with token
- Always test notebook execution with: `jupyter nbconvert --to python --execute [notebook].ipynb`

### Educational Content Structure
Each learning step typically includes:
- **Lesson file** (lessons/): Mathematical theory and explanations
- **Code file(s)** (code/): Complete implementations from scratch  
- **Exercise file** (exercises/): Practice problems with detailed solutions
- **Notebook file** (notebooks/): Interactive visualizations and experiments

## Dependencies and Versions

The repository uses these main packages (see requirements.txt):
- numpy>=1.21.0 (core mathematical operations)
- matplotlib>=3.5.0 (plotting and visualization)
- scipy>=1.7.0 (advanced mathematical functions)
- scikit-learn>=1.0.0 (built-in datasets and comparison implementations)
- pandas>=1.3.0 (data manipulation)
- jupyter>=1.0.0 (interactive notebooks)
- torch>=1.12.0 (deep learning framework)
- plotly>=5.0.0 (interactive visualizations)
- sympy>=1.9.0 (symbolic mathematics)

## Performance Expectations

### Timing Guidelines (NEVER CANCEL these operations)
- **pip install -r requirements.txt**: 45-180 seconds (set 300+ second timeout)
- **jupyter lab startup**: ~10 seconds
- **PCA code execution**: ~4 seconds
- **Basic import tests**: instant (<1 second)
- **Virtual environment creation**: instant (<5 seconds)

### Memory and Resource Usage
- Typical memory usage: 200-500MB for basic operations
- PyTorch may use additional GPU memory if available
- Jupyter Lab uses ~100MB base memory
- Large datasets or visualizations may require more memory

## Development Workflow

1. **Always activate virtual environment first**: `source venv/bin/activate`
2. **Test existing working code before changes**: `python code/01_pca.py`
3. **Make incremental changes and test frequently**
4. **Validate with manual scenarios after each change**
5. **Use Jupyter Lab for interactive development and visualization**
6. **Follow educational code style**: clear comments, mathematical explanations, step-by-step breakdowns
7. **Use .gitignore to exclude build artifacts**: venv/, *.png, __pycache__/, etc. are already configured

## Emergency Troubleshooting

### If imports fail:
```bash
# Check environment activation
which python  # Should point to venv
pip list      # Verify packages installed

# Reinstall if needed (NEVER CANCEL - takes 45-180 seconds)
pip install --timeout=300 --retries=5 -r requirements.txt

# If network timeouts persist, install core packages individually:
pip install numpy matplotlib scipy scikit-learn pandas
pip install jupyter torch torchvision plotly seaborn sympy
```

### If Jupyter won't start:
```bash
# Install explicitly
pip install jupyter jupyterlab
jupyter lab --version  # Verify installation
```

### If code has plotting issues:
```python
# Add at top of Python files
import matplotlib
matplotlib.use('Agg')  # For non-interactive plots
```