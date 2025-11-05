import warnings
import glob

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from optax import adam, apply_updates

import reciprocalspaceship as rs
import numpy as np

key = jax.random.PRNGKey(10)
warnings.filterwarnings("ignore")


def soft_threshold(w, lambda_threshold):
    """
    Apply soft thresholding (proximal operator for L1 norm).
    This allows weights to go exactly to 0.

    Args:
        w: weights
        lambda_threshold: threshold value

    Returns:
        Thresholded weights
    """
    return jnp.sign(w) * jnp.maximum(jnp.abs(w) - lambda_threshold, 0.0)


def hard_threshold(w, threshold=0.01):
    """
    Apply hard thresholding - set weights below threshold to exactly 0.

    Args:
        w: weights
        threshold: cutoff value

    Returns:
        Thresholded weights
    """
    return jnp.where(jnp.abs(w) < threshold, 0.0, w)


def get_mesh_sharding(n_devices=None):
    """
    Create a mesh for sharding across available GPUs.

    Args:
        n_devices: Number of devices to use (None = use all available)

    Returns:
        Mesh object for sharding
    """
    devices = jax.devices("gpu") if jax.devices("gpu") else jax.devices("cpu")
    if n_devices:
        devices = devices[:n_devices]

    print(f"Using {len(devices)} device(s): {[d.device_kind for d in devices]}")

    # Create 1D mesh along device axis
    mesh_devices = mesh_utils.create_device_mesh((len(devices),))
    mesh = Mesh(mesh_devices, axis_names=("data",))

    return mesh


def pearson_cc(x, y):
    mean_x = jnp.nanmean(x)
    mean_y = jnp.nanmean(y)
    mean_xy = jnp.nanmean(x * y)
    mean_x2 = jnp.nanmean(x * x)
    mean_y2 = jnp.nanmean(y * y)

    numerator = mean_xy - mean_x * mean_y
    denominator = jnp.sqrt(mean_x2 - mean_x**2) * jnp.sqrt(mean_y2 - mean_y**2)

    return numerator / denominator


def compute_xprime(w, F_array):
    # n_timepoints = w.shape[0] # NOTE: commented because if weights are constrained to sum to 1, this should not be in the

    weighted_F2 = jnp.einsum("t,t...->...", w, jnp.abs(F_array) ** 2)  # / n_timepoints
    weighted_F = jnp.einsum("t,t...->...", w, F_array)  # / n_timepoints

    xprime = weighted_F2 - jnp.abs(weighted_F) ** 2

    return xprime


def objective(w, F_array, y, lambda_l1=0.0, lambda_l2=0.0):
    xprime = compute_xprime(w, F_array)
    xprime_flat = xprime.flatten()
    y_flat = y.flatten()

    cc = pearson_cc(xprime_flat, y_flat)

    l2_penalty = lambda_l2 * jnp.mean((w - 1.0) ** 2)
    l1_penalty = lambda_l1 * jnp.mean(jnp.abs(w - 1.0))

    return cc - l2_penalty - l1_penalty


def optimize_weights(
    F_array,
    y,
    n_steps,
    step_size,
    batch_size=100000,
    lambda_l1=0.0,
    lambda_l2=0.0,
    use_proximal=False,
    proximal_lambda=0.01,
    use_sigmoid=True,
    hard_threshold_final=None,
):
    """
    Optimize weights for dataset combination.

    Args:
        F_array: Array of structure factors, shape (n_datasets, n_reflections, ...)
        y: Target intensities, shape (n_reflections,)
        n_steps: Number of optimization steps
        step_size: Learning rate for Adam optimizer
        batch_size: Number of reflections to use in batched gradient step
        lambda_l1: L1 regularization strength in objective
        lambda_l2: L2 regularization strength in objective
        use_proximal: If True, use proximal gradient descent with soft thresholding after each step
        proximal_lambda: Threshold value for soft thresholding (only used if use_proximal=True)
        use_sigmoid: If True, use sigmoid parameterization (weights in [0,1]). If False, use softmax (weights sum to 1)
        hard_threshold_final: If not None, apply hard thresholding at this value after optimization

    Returns:
        Optimized weights
    """
    n_timepoints = F_array.shape[0]
    n_reflections = y.shape[0]

    # Initialize parameters
    if use_sigmoid:
        u = jnp.full((n_timepoints,), -jax.numpy.log(n_timepoints - 1))
        transform_fn = jax.nn.sigmoid
    else:
        # Use softmax parameterization - weights sum to 1 and can go to 0
        u = jnp.zeros((n_timepoints,))

        def transform_fn(x):
            return jax.nn.softmax(x)

    print("Initial weights:", transform_fn(u))
    print("Initial CC:", objective(transform_fn(u), F_array, y, lambda_l1, lambda_l2))

    optimizer = adam(step_size)
    params = {"u": u}
    opt_state = optimizer.init(params)

    grad_fn = jit(
        grad(
            lambda params, F_array, y: -objective(
                transform_fn(params["u"]), F_array, y, lambda_l1, lambda_l2
            )
        )
    )

    last_cc = -jnp.inf

    for step in range(n_steps):
        # Full gradient step
        g = grad_fn(params, F_array, y)
        updates, opt_state = optimizer.update(g, opt_state)
        params = apply_updates(params, updates)

        # Apply proximal operator (soft thresholding) if requested
        if use_proximal and not use_sigmoid:
            # Only apply proximal operator when using unconstrained parameterization
            params["u"] = soft_threshold(params["u"], proximal_lambda)

        # Batched gradient step
        batch_key = jax.random.fold_in(key, step)
        batch_indices = jax.random.choice(
            batch_key, n_reflections, shape=(batch_size,), replace=False
        )

        F_batch = F_array[:, batch_indices]
        y_batch = y[batch_indices]

        g = grad_fn(params, F_batch, y_batch)
        updates, opt_state = optimizer.update(g, opt_state)
        params = apply_updates(params, updates)

        # Apply proximal operator again after batch step
        if use_proximal and not use_sigmoid:
            params["u"] = soft_threshold(params["u"], proximal_lambda)

        # Logging and convergence check
        if step % 10 == 0:
            w = transform_fn(params["u"])
            cc = objective(w, F_array, y, lambda_l1, lambda_l2)
            n_nonzero = jnp.sum(w > 1e-6)
            print(
                f"Step {step}: Objective = {cc:.6f}, Non-zero weights: {n_nonzero}/{n_timepoints}"
            )

            if jnp.allclose(cc, last_cc, atol=1e-5):
                print(f"Converged at step {step}")
                break
            last_cc = cc

    w = transform_fn(params["u"])

    # Apply hard thresholding if requested
    if hard_threshold_final is not None:
        print(f"Applying hard threshold at {hard_threshold_final}")
        w_before = w
        w = hard_threshold(w, hard_threshold_final)
        n_zeroed = jnp.sum((w_before > 0) & (w == 0))
        print(f"Set {n_zeroed} weights to exactly 0")

    return w


def subsample_reflections(datasets, y, subsample_fraction=1.0, random_seed=42):
    """
    Subsample reflections to reduce memory usage.

    Args:
        datasets: List of numpy arrays (one per MTZ)
        y: Ground truth intensities
        subsample_fraction: Fraction of reflections to keep (0.0 to 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        subsampled_datasets: List of subsampled arrays
        subsampled_y: Subsampled ground truth
    """
    if subsample_fraction >= 1.0:
        return datasets, y

    n_reflections = y.shape[0]
    n_keep = int(n_reflections * subsample_fraction)

    print(
        f"Subsampling reflections: {n_reflections} -> {n_keep} ({subsample_fraction * 100:.1f}%)"
    )

    # Use numpy for reproducible random sampling
    rng = np.random.RandomState(random_seed)
    indices = rng.choice(n_reflections, size=n_keep, replace=False)
    indices = np.sort(indices)  # Keep sorted for better memory access

    subsampled_datasets = [d[indices] for d in datasets]
    subsampled_y = y[indices]

    return subsampled_datasets, subsampled_y


def load_mtzs_with_sharding(
    mtz_files, nas, y, use_sharding=False, subsample_fraction=1.0
):
    """
    Load MTZ files with optional sharding across GPUs and subsampling.

    Args:
        mtz_files: List of MTZ file paths
        nas: NaN mask from ground truth
        y: Ground truth intensities
        use_sharding: If True, shard data across available GPUs
        subsample_fraction: Fraction of reflections to keep

    Returns:
        F_array: Stacked structure factor array (potentially subsampled and sharded)
        y: Ground truth intensities (potentially subsampled)
        mesh: Sharding mesh (None if use_sharding=False)
    """
    datasets = []

    for mtz in mtz_files:
        if "sqrtIdiffuse" in mtz:
            continue
        print("Reading:", mtz.split("/")[-1])
        dataset = (
            rs.read_mtz(mtz)
            .expand_to_p1()[~nas.sqrtIdiff]
            .to_structurefactor("FMODEL", "PHIFMODEL")
            .to_numpy()
        )
        datasets.append(dataset)

    if not datasets:
        raise ValueError("No MTZ files loaded!")

    if subsample_fraction < 1.0:
        # Convert F_array back to list for subsampling
        datasets, y = subsample_reflections(
            datasets, y, subsample_fraction=subsample_fraction
        )

    F_array = jnp.stack(datasets, axis=0)
    print(f"F_array shape: {F_array.shape}")
    print(f"F_array size: {F_array.nbytes / 1e9:.2f} GB")

    if use_sharding:
        mesh = get_mesh_sharding()

        # Create sharding specification: shard along dataset axis
        sharding = NamedSharding(mesh, PartitionSpec("data", None))
        F_array = jax.device_put(F_array, sharding)

        print(f"Data sharded across devices: {F_array.sharding}")
    else:
        mesh = None

    return F_array, y, mesh


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Memory reduction options
    USE_SHARDING = True  # Set to True to use multi-GPU sharding
    SUBSAMPLE_FRACTION = 1.0  # Fraction of reflections to use (e.g., 0.5 = 50%)

    # Sparse optimization options
    USE_PROXIMAL = True  # Set to True to use proximal gradient descent
    PROXIMAL_LAMBDA = 0.01  # Soft threshold value for proximal operator
    USE_SIGMOID = True  # If False, uses softmax (allows weights to reach 0)
    HARD_THRESHOLD_FINAL = (
        0.01  # Apply hard threshold after optimization (None to disable)
    )

    # Regularization
    LAMBDA_L1 = 0.1  # L1 regularization strength
    LAMBDA_L2 = 0.0  # L2 regularization strength

    # Optimization parameters
    N_STEPS = 500
    STEP_SIZE = 0.05
    BATCH_SIZE = 100000
    # ===================================

    print("=" * 60)
    print("GPU CALCULATION OPTIMIZATION")
    print("=" * 60)
    print("Configuration:")
    print(f"  Multi-GPU sharding: {USE_SHARDING}")
    print(f"  Subsample fraction: {SUBSAMPLE_FRACTION}")
    print(f"  Use proximal (soft threshold): {USE_PROXIMAL}")
    print(f"  Use sigmoid parameterization: {USE_SIGMOID}")
    print(f"  Hard threshold final: {HARD_THRESHOLD_FINAL}")
    print(f"  L1 regularization: {LAMBDA_L1}")
    print(f"  L2 regularization: {LAMBDA_L2}")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_dataset = rs.read_mtz(
        "diffUSE_CC_opt_test/sqrtIdiffuse_ground_truth.mtz"
    ).expand_to_p1()
    nas = gt_dataset.isna()
    gt_dataset = gt_dataset[~nas.sqrtIdiff]
    print("Ground truth:\n", gt_dataset.head())
    y = gt_dataset.sqrtIdiff.to_numpy() ** 2
    print("Ground truth obs shape: ", y.shape)

    # Load MTZ files
    print("\nLoading MTZ files...")
    mtzs = glob.glob("diffUSE_CC_opt_test/*.mtz")
    mtz_files = [mtz for mtz in mtzs if "sqrtIdiffuse" not in mtz]

    # Memory-efficient loading with optional sharding
    F_array, y, mesh = load_mtzs_with_sharding(
        mtz_files,
        nas,
        y,
        use_sharding=USE_SHARDING,
        subsample_fraction=1.0,  # Subsampling done separately for now
    )

    print("F_array shape:", F_array.shape)
    print(f"F_array size: {F_array.nbytes / 1e9:.2f} GB")

    # Optimize weights with sparse optimization
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)

    weights = optimize_weights(
        F_array,
        y,
        n_steps=N_STEPS,
        step_size=STEP_SIZE,
        batch_size=BATCH_SIZE,
        lambda_l1=LAMBDA_L1,
        lambda_l2=LAMBDA_L2,
        use_proximal=USE_PROXIMAL,
        proximal_lambda=PROXIMAL_LAMBDA,
        use_sigmoid=USE_SIGMOID,
        hard_threshold_final=HARD_THRESHOLD_FINAL,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Datasets:", [mtz.split("/")[-1] for mtz in mtz_files])
    print("\nFinal weights:")
    for i, (mtz, w) in enumerate(zip(mtz_files, weights)):
        status = "ZERO" if w == 0 else f"{w:.6f}"
        print(f"  {i:2d}. {mtz.split('/')[-1]:40s} -> {status}")

    n_zero = jnp.sum(weights == 0)
    n_near_zero = jnp.sum((weights > 0) & (weights < 0.01))
    print("\nWeight statistics:")
    print(f"  Exactly zero: {n_zero}/{len(weights)}")
    print(f"  Near zero (<0.01): {n_near_zero}/{len(weights)}")
    print(f"  Non-zero (>=0.01): {jnp.sum(weights >= 0.01)}/{len(weights)}")
    print(f"  Mean weight: {jnp.mean(weights):.6f}")
    print(f"  Max weight: {jnp.max(weights):.6f}")

    xprime = compute_xprime(weights, F_array)
    final_cc = pearson_cc(xprime, y)
    print(f"\nFinal Pearson CC: {final_cc:.6f}")
    print("=" * 60)
