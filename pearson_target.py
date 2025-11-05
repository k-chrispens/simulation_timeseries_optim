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


def smooth_objective(w, F_array, y, lambda_l2=0.0):
    """
    Smooth part of the objective (without L1 penalty).
    Used for true proximal gradient descent.

    Args:
        w: weights
        F_array: structure factors
        y: ground truth intensities
        lambda_l2: L2 regularization strength

    Returns:
        Smooth objective value (Pearson CC - L2 penalty)
    """
    xprime = compute_xprime(w, F_array)
    xprime_flat = xprime.flatten()
    y_flat = y.flatten()

    cc = pearson_cc(xprime_flat, y_flat)
    l2_penalty = lambda_l2 * jnp.mean((w - 1.0) ** 2)

    return cc - l2_penalty


def objective(w, F_array, y, lambda_l1=0.0, lambda_l2=0.0):
    """
    Full objective including L1 penalty.
    Used for evaluation and non-proximal optimization.
    """
    smooth_obj = smooth_objective(w, F_array, y, lambda_l2)
    l1_penalty = lambda_l1 * jnp.mean(jnp.abs(w - 1.0))

    return smooth_obj - l1_penalty


def optimize_weights(
    F_array,
    y,
    n_steps,
    step_size,
    batch_size=100000,
    lambda_l1=0.0,
    lambda_l2=0.0,
    use_proximal=False,
    use_sigmoid=True,
    hard_threshold_final=None,
):
    """
    Optimize weights for dataset combination.

    Args:
        F_array: Array of structure factors, shape (n_datasets, n_reflections, ...)
        y: Target intensities, shape (n_reflections,)
        n_steps: Number of optimization steps
        step_size: Learning rate for optimizer
        batch_size: Number of reflections to use in batched gradient step
        lambda_l1: L1 regularization strength
        lambda_l2: L2 regularization strength
        use_proximal: If True, use TRUE proximal gradient descent (gradient on smooth part only,
                      then apply proximal operator for L1)
        use_sigmoid: If True, use sigmoid parameterization (weights in [0,1]).
                     If False, use raw parameterization (required for proximal)
        hard_threshold_final: If not None, apply hard thresholding at this value after optimization

    Returns:
        Optimized weights
    """
    n_timepoints = F_array.shape[0]
    n_reflections = y.shape[0]

    # Initialize parameters
    if use_sigmoid:
        if use_proximal:
            raise ValueError("Cannot use proximal gradient with sigmoid parameterization. Set use_sigmoid=False.")
        u = jnp.full((n_timepoints,), -jax.numpy.log(n_timepoints - 1))
        transform_fn = jax.nn.sigmoid
    else:
        # Direct parameterization - weights are the parameters
        # Initialize to uniform
        u = jnp.ones((n_timepoints,)) / n_timepoints
        transform_fn = lambda x: x  # Identity

    print("Initial weights:", transform_fn(u))

    # For proximal, evaluate on smooth objective; for regular, use full objective
    if use_proximal:
        initial_obj = smooth_objective(transform_fn(u), F_array, y, lambda_l2)
        print(f"Initial smooth objective: {initial_obj:.6f}")
    else:
        initial_obj = objective(transform_fn(u), F_array, y, lambda_l1, lambda_l2)
        print(f"Initial objective: {initial_obj:.6f}")

    optimizer = adam(step_size)
    params = {"u": u}
    opt_state = optimizer.init(params)

    # Create gradient function based on whether we're using proximal or not
    if use_proximal:
        # TRUE PROXIMAL: gradient of smooth part only (no L1)
        grad_fn = jit(
            grad(
                lambda params, F_array, y: -smooth_objective(
                    params["u"], F_array, y, lambda_l2
                )
            )
        )
        print("Using TRUE proximal gradient descent (gradient on smooth objective only)")
    else:
        # REGULAR: gradient of full objective (including L1)
        grad_fn = jit(
            grad(
                lambda params, F_array, y: -objective(
                    transform_fn(params["u"]), F_array, y, lambda_l1, lambda_l2
                )
            )
        )
        print("Using standard gradient descent (gradient on full objective)")

    last_cc = -jnp.inf

    for step in range(n_steps):
        # Full gradient step
        g = grad_fn(params, F_array, y)
        updates, opt_state = optimizer.update(g, opt_state)
        params = apply_updates(params, updates)

        # TRUE PROXIMAL: Apply proximal operator for L1 with proper scaling
        if use_proximal:
            # The proximal operator accounts for the L1 penalty
            # Scaling by step_size is crucial for convergence
            params["u"] = soft_threshold(params["u"], step_size * lambda_l1)

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
        if use_proximal:
            params["u"] = soft_threshold(params["u"], step_size * lambda_l1)

        # Logging and convergence check
        if step % 10 == 0:
            w = transform_fn(params["u"])
            # Always evaluate on full objective for comparison
            cc = objective(w, F_array, y, lambda_l1, lambda_l2)
            n_nonzero = jnp.sum(w > 1e-6)
            n_zero = jnp.sum(w == 0)
            print(
                f"Step {step}: Objective = {cc:.6f}, Exact zeros: {n_zero}/{n_timepoints}, Non-zero: {n_nonzero}/{n_timepoints}"
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


def optimize_weights_batched_hkl(
    datasets_cpu,
    y_cpu,
    n_steps,
    step_size,
    hkl_batch_size=10000,
    lambda_l1=0.0,
    lambda_l2=0.0,
    use_proximal=False,
    use_sigmoid=True,
    hard_threshold_final=None,
    use_sharding=False,
):
    n_timepoints = len(datasets_cpu)
    n_reflections = y_cpu.shape[0]

    print("Batched HKL optimization:")
    print(f"  Total reflections: {n_reflections}")
    print(f"  HKL batch size: {hkl_batch_size}")
    print(f"  Datasets: {n_timepoints}")
    print(f"  Multi-GPU sharding: {use_sharding}")

    mesh = None
    sharding_spec = None
    if use_sharding:
        mesh = get_mesh_sharding()
        sharding_spec = NamedSharding(mesh, PartitionSpec("data", None))

    # Initialize parameters
    if use_sigmoid:
        if use_proximal:
            raise ValueError("Cannot use proximal gradient with sigmoid parameterization. Set use_sigmoid=False.")
        u = jnp.full((n_timepoints,), -jax.numpy.log(n_timepoints - 1))
        transform_fn = jax.nn.sigmoid
    else:
        # Direct parameterization
        u = jnp.ones((n_timepoints,)) / n_timepoints
        def transform_fn(x):
            return x  # Identity

    print("Initial weights:", transform_fn(u))

    optimizer = adam(step_size)
    params = {"u": u}
    opt_state = optimizer.init(params)

    # Create gradient function
    if use_proximal:
        grad_fn = jit(
            grad(
                lambda params, F_array, y: -smooth_objective(
                    params["u"], F_array, y, lambda_l2
                )
            )
        )
        print("Using TRUE proximal gradient descent (batched HKL)")
    else:
        grad_fn = jit(
            grad(
                lambda params, F_array, y: -objective(
                    transform_fn(params["u"]), F_array, y, lambda_l1, lambda_l2
                )
            )
        )
        print("Using standard gradient descent (batched HKL)")

    last_cc = -jnp.inf

    eval_batch_size = min(50000, n_reflections)
    eval_indices = np.arange(eval_batch_size)
    F_eval_np = np.stack([d[eval_indices] for d in datasets_cpu], axis=0)
    if use_sharding:
        F_eval = jax.device_put(F_eval_np, sharding_spec)
    else:
        F_eval = jnp.array(F_eval_np)
    y_eval = jnp.array(y_cpu[eval_indices])

    for step in range(n_steps):
        batch_key = jax.random.fold_in(key, step)
        batch_indices = jax.random.choice(
            batch_key, n_reflections, shape=(hkl_batch_size,), replace=False
        )

        batch_indices_np = np.array(batch_indices)

        F_batch_np = np.stack([d[batch_indices_np] for d in datasets_cpu], axis=0)
        if use_sharding:
            F_batch = jax.device_put(F_batch_np, sharding_spec)
        else:
            F_batch = jnp.array(F_batch_np)
        y_batch = jnp.array(y_cpu[batch_indices_np])

        # Gradient step
        g = grad_fn(params, F_batch, y_batch)
        updates, opt_state = optimizer.update(g, opt_state)
        params = apply_updates(params, updates)

        # Apply proximal operator if requested
        if use_proximal:
            params["u"] = soft_threshold(params["u"], step_size * lambda_l1)

        # Logging and convergence check
        if step % 10 == 0:
            w = transform_fn(params["u"])
            # Evaluate on eval subset
            cc = objective(w, F_eval, y_eval, lambda_l1, lambda_l2)
            n_nonzero = jnp.sum(w > 1e-6)
            n_zero = jnp.sum(w == 0)
            print(
                f"Step {step}: Objective = {cc:.6f}, Exact zeros: {n_zero}/{n_timepoints}, Non-zero: {n_nonzero}/{n_timepoints}"
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


def load_mtzs_cpu_only(mtz_files, nas, y, subsample_fraction=1.0):
    """
    Load MTZ files and keep them in system memory (for batched HKL loading).

    Args:
        mtz_files: List of MTZ file paths
        nas: NaN mask from ground truth
        y: Ground truth intensities
        subsample_fraction: Fraction of reflections to keep

    Returns:
        datasets: List of numpy arrays (kept in system memory)
        y: Ground truth intensities (potentially subsampled)
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
        datasets, y = subsample_reflections(
            datasets, y, subsample_fraction=subsample_fraction
        )

    print(f"Loaded {len(datasets)} datasets to system memory")
    print(f"Reflection shape: {datasets[0].shape}")
    total_size = sum(d.nbytes for d in datasets) / 1e9
    print(f"Total data size: {total_size:.2f} GB (in system memory)")

    return datasets, y


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
    USE_BATCHED_HKL = True
    USE_SHARDING = True
    SUBSAMPLE_FRACTION = 1.0
    HKL_BATCH_SIZE = 20000

    USE_PROXIMAL = True
    USE_SIGMOID = False
    HARD_THRESHOLD_FINAL = 0.01

    LAMBDA_L1 = 0.1
    LAMBDA_L2 = 0.0

    N_STEPS = 500
    STEP_SIZE = 0.05
    BATCH_SIZE = 100000

    print("=" * 60)
    print("GPU CALCULATION OPTIMIZATION")
    print("=" * 60)
    print("Configuration:")
    print(f"  Memory strategy: {'Batched HKL' if USE_BATCHED_HKL else 'Full GPU'}")
    print(f"  Multi-GPU sharding: {USE_SHARDING}")
    if USE_BATCHED_HKL:
        print(f"  HKL batch size: {HKL_BATCH_SIZE}")
    else:
        print(f"  Subsample fraction: {SUBSAMPLE_FRACTION}")
    print(f"  Proximal gradient: {USE_PROXIMAL}")
    print(f"  L1: {LAMBDA_L1}, L2: {LAMBDA_L2}")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_dataset = rs.read_mtz(
        "sqrtIdiff_observed.mtz"
    ).expand_to_p1()
    nas = gt_dataset.isna()
    gt_dataset = gt_dataset[~nas.sqrtIdiff]
    print("Ground truth:\n", gt_dataset.head())
    y = gt_dataset.sqrtIdiff.to_numpy() ** 2
    print("Ground truth obs shape: ", y.shape)

    # Load MTZ files
    print("\nLoading MTZ files...")
    mtzs = glob.glob("mtz1.8A/*.mtz")
    mtz_files = [mtz for mtz in mtzs if "sqrtIdiffuse" not in mtz]

    # Optimize weights with sparse optimization
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)

    if USE_BATCHED_HKL:
        datasets_cpu, y = load_mtzs_cpu_only(
            mtz_files,
            nas,
            y,
            subsample_fraction=SUBSAMPLE_FRACTION,
        )

        weights = optimize_weights_batched_hkl(
            datasets_cpu,
            y,
            n_steps=N_STEPS,
            step_size=STEP_SIZE,
            hkl_batch_size=HKL_BATCH_SIZE,
            lambda_l1=LAMBDA_L1,
            lambda_l2=LAMBDA_L2,
            use_proximal=USE_PROXIMAL,
            use_sigmoid=USE_SIGMOID,
            hard_threshold_final=HARD_THRESHOLD_FINAL,
            use_sharding=USE_SHARDING,
        )

        print("\nFinal evaluation...")
        eval_size = min(50000, y.shape[0])
        F_eval = jnp.stack([d[:eval_size] for d in datasets_cpu], axis=0)
        y_eval = jnp.array(y[:eval_size])

    else:
        F_array, y, mesh = load_mtzs_with_sharding(
            mtz_files,
            nas,
            y,
            use_sharding=USE_SHARDING,
            subsample_fraction=SUBSAMPLE_FRACTION,
        )

        print("F_array shape:", F_array.shape)
        print(f"F_array size: {F_array.nbytes / 1e9:.2f} GB")

        weights = optimize_weights(
            F_array,
            y,
            n_steps=N_STEPS,
            step_size=STEP_SIZE,
            batch_size=BATCH_SIZE,
            lambda_l1=LAMBDA_L1,
            lambda_l2=LAMBDA_L2,
            use_proximal=USE_PROXIMAL,
            use_sigmoid=USE_SIGMOID,
            hard_threshold_final=HARD_THRESHOLD_FINAL,
        )

        F_eval = F_array
        y_eval = y

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

    # Compute final correlation on evaluation data
    xprime = compute_xprime(weights, F_eval)
    final_cc = pearson_cc(xprime, y_eval)
    print(f"\nFinal Pearson CC: {final_cc:.6f}")
    if USE_BATCHED_HKL:
        print(f"  (evaluated on {eval_size} reflections)")
    print("=" * 60)
