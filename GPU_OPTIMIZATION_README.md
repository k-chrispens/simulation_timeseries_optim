# GPU Calculation Improvements

This document explains the improvements made to address GPU memory issues and enable exact zero weights during optimization.

## Problems Addressed

### 1. GPU Memory Overload
**Issue**: When loading large sets of MTZ files, the `jnp.stack()` command would exhaust GPU memory, especially with many datasets and reflections.

**Solutions Implemented**:
- **Reflection Subsampling**: Reduce memory by using a fraction of reflections
- **Multi-GPU Sharding**: Distribute data across multiple GPUs using JAX sharding
- **Memory-Efficient Loading**: Improved data loading pipeline

### 2. Weights Cannot Reach Zero
**Issue**: With sigmoid parameterization, weights are constrained to (0, 1) and can never reach exactly 0, even with L1 regularization.

**Solutions Implemented**:
- **Softmax Parameterization**: Allows weights to naturally go to 0 while maintaining sum-to-1 constraint
- **Proximal Gradient Descent**: Soft thresholding operator for L1 sparsity
- **Hard Thresholding**: Post-optimization thresholding to force small weights to exactly 0

---

## New Features

### Memory Reduction Options

#### 1. Reflection Subsampling
Randomly sample a subset of reflections to reduce memory usage:

```python
SUBSAMPLE_FRACTION = 0.5  # Use 50% of reflections
```

**Benefits**:
- Linear reduction in memory usage
- Maintains statistical properties
- Faster optimization

**Trade-offs**:
- Slightly reduced signal
- May need more regularization

#### 2. Multi-GPU Sharding
Distribute computation across multiple GPUs:

```python
USE_SHARDING = True  # Enable multi-GPU sharding
```

**Benefits**:
- Scales to larger datasets
- Better GPU utilization
- Automatic load balancing

**Requirements**:
- Multiple GPUs available
- JAX with GPU support

### Sparse Optimization Options

#### 1. Softmax Parameterization
Use softmax instead of sigmoid to allow weights to reach 0:

```python
USE_SIGMOID = False  # Use softmax (allows exact zeros)
```

**Behavior**:
- Weights sum to 1
- Can reach exactly 0
- Individual weights unconstrained

**vs. Sigmoid**:
- Sigmoid: weights in (0, 1), cannot be 0
- Softmax: weights sum to 1, can be 0

#### 2. Proximal Gradient Descent
Apply soft thresholding after each gradient step:

```python
USE_PROXIMAL = True
PROXIMAL_LAMBDA = 0.01  # Threshold strength
```

**Mechanism**:
- After gradient update: `w = sign(w) * max(|w| - λ, 0)`
- Induces sparsity naturally
- Standard approach for L1 optimization

**When to use**:
- Want exact zeros during optimization
- Strong sparsity desired
- Works best with `USE_SIGMOID = False`

#### 3. Hard Thresholding
Force small weights to exactly 0 after optimization:

```python
HARD_THRESHOLD_FINAL = 0.01  # Set weights < 0.01 to 0
```

**Behavior**:
- Applied after optimization completes
- Simple post-processing step
- Guaranteed exact zeros

**When to use**:
- Want to prune low-weight datasets
- Cleaner final results
- Works with any parameterization

---

## Configuration Guide

### Recommended Settings by Use Case

#### Case 1: Large Dataset, GPU Memory Limited
**Goal**: Fit large dataset into GPU memory

```python
# Memory reduction
USE_SHARDING = True              # If multiple GPUs available
SUBSAMPLE_FRACTION = 0.7         # Use 70% of reflections

# Standard optimization
USE_SIGMOID = False
USE_PROXIMAL = False
HARD_THRESHOLD_FINAL = 0.01      # Clean up small weights
```

#### Case 2: Maximum Sparsity (Most Exact Zeros)
**Goal**: Identify minimal set of important datasets

```python
# Aggressive sparsity
USE_SIGMOID = False              # Allow zeros
USE_PROXIMAL = True              # Soft threshold during optimization
PROXIMAL_LAMBDA = 0.02           # Aggressive threshold
HARD_THRESHOLD_FINAL = 0.01      # Final cleanup

# Strong regularization
LAMBDA_L1 = 0.2                  # High L1 penalty
```

#### Case 3: Balanced Approach (Recommended Starting Point)
**Goal**: Good performance with moderate sparsity

```python
# Memory (adjust as needed)
USE_SHARDING = False
SUBSAMPLE_FRACTION = 1.0         # Use all data

# Moderate sparsity
USE_SIGMOID = False              # Allow zeros
USE_PROXIMAL = False             # No soft threshold
HARD_THRESHOLD_FINAL = 0.01      # Clean small weights at end

# Moderate regularization
LAMBDA_L1 = 0.1
```

---

## Parameter Reference

### Memory Reduction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_SHARDING` | bool | `False` | Enable multi-GPU data sharding |
| `SUBSAMPLE_FRACTION` | float | `1.0` | Fraction of reflections to use (0.0-1.0) |

### Sparse Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_SIGMOID` | bool | `False` | Use sigmoid (True) or softmax (False) |
| `USE_PROXIMAL` | bool | `False` | Apply proximal operator (soft threshold) |
| `PROXIMAL_LAMBDA` | float | `0.01` | Soft threshold strength |
| `HARD_THRESHOLD_FINAL` | float/None | `0.01` | Hard threshold after optimization |

### Regularization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LAMBDA_L1` | float | `0.1` | L1 penalty strength in objective |
| `LAMBDA_L2` | float | `0.0` | L2 penalty strength in objective |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N_STEPS` | int | `500` | Maximum optimization iterations |
| `STEP_SIZE` | float | `0.05` | Adam learning rate |
| `BATCH_SIZE` | int | `100000` | Reflections per batch gradient step |

---

## Technical Details

### Soft Thresholding (Proximal Operator)
The soft thresholding operator is the proximal operator for the L1 norm:

```
prox_λ(w) = sign(w) * max(|w| - λ, 0)
```

This shrinks weights toward 0, with weights smaller than λ going to exactly 0.

### Hard Thresholding
Simple cutoff:

```
threshold_τ(w) = w  if |w| >= τ
                 0  if |w| < τ
```

### JAX Sharding
Data is distributed across GPUs along the dataset axis:
- Each GPU handles a subset of datasets
- Reflections are replicated across GPUs
- Automatic gradient aggregation

---

## Performance Tips

### Memory Usage
1. **Monitor GPU memory**: Use `nvidia-smi` to check usage
2. **Start conservative**: Begin with `SUBSAMPLE_FRACTION = 0.5`
3. **Increase gradually**: Raise fraction until memory is full
4. **Use sharding**: If multiple GPUs available, enable sharding

### Sparsity
1. **Start with hard thresholding**: Easiest to understand and tune
2. **Add proximal if needed**: For more aggressive sparsity
3. **Tune threshold values**: Higher = more sparsity, but may hurt performance
4. **Check convergence**: Ensure optimization still converges with strong regularization

### Optimization Speed
1. **Batch size**: Larger = faster per iteration, but may reduce convergence quality
2. **Learning rate**: Lower if optimization is unstable
3. **Early stopping**: Optimization stops automatically when converged

---

## Output Interpretation

The improved version provides detailed statistics:

```
Weight statistics:
  Exactly zero: 15/100        # Datasets completely removed
  Near zero (<0.01): 10/100   # Datasets with very small weight
  Non-zero (>=0.01): 75/100   # Datasets contributing significantly
  Mean weight: 0.012000
  Max weight: 0.089000

Final Pearson CC: 0.856234    # Correlation with ground truth
```

### What to look for:
- **High sparsity + high CC**: Good dataset selection
- **Low sparsity + high CC**: Many datasets needed
- **High sparsity + low CC**: Over-regularization, reduce λ
- **Many near-zero weights**: Consider increasing `HARD_THRESHOLD_FINAL`

---

## Migration from Old Version

**Old code**:
```python
weights = optimize_weights(F_array, y, n_steps=500, step_size=0.05,
                          lambda_l1=0.1, lambda_l2=0.)
```

**New code (equivalent behavior)**:
```python
weights = optimize_weights(F_array, y, n_steps=500, step_size=0.05,
                          lambda_l1=0.1, lambda_l2=0.,
                          use_sigmoid=False,  # Changed default
                          hard_threshold_final=0.01)  # New feature
```

The new default uses softmax instead of sigmoid, which allows weights to reach 0.

---

## Troubleshooting

### GPU Out of Memory
1. Enable subsampling: `SUBSAMPLE_FRACTION = 0.5`
2. Reduce batch size: `BATCH_SIZE = 50000`
3. Enable sharding if multiple GPUs: `USE_SHARDING = True`

### Weights Not Going to Zero
1. Disable sigmoid: `USE_SIGMOID = False`
2. Enable hard thresholding: `HARD_THRESHOLD_FINAL = 0.01`
3. Try proximal: `USE_PROXIMAL = True`
4. Increase L1: `LAMBDA_L1 = 0.2`

### Optimization Not Converging
1. Reduce learning rate: `STEP_SIZE = 0.01`
2. Reduce regularization: `LAMBDA_L1 = 0.05`
3. Disable proximal: `USE_PROXIMAL = False`
4. Increase steps: `N_STEPS = 1000`

### Poor Final CC
1. Reduce sparsity: Lower `LAMBDA_L1`, `PROXIMAL_LAMBDA`, `HARD_THRESHOLD_FINAL`
2. Use more data: Increase `SUBSAMPLE_FRACTION`
3. Check ground truth quality
4. Verify MTZ files are correctly formatted
