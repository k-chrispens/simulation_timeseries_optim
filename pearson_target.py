import warnings
import glob

import jax
import jax.numpy as jnp
from jax import grad, jit
from optax import adam, apply_updates

import reciprocalspaceship as rs

key = jax.random.PRNGKey(10)
warnings.filterwarnings("ignore")


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


def objective(w, F_array, y, lambda_l1=0., lambda_l2=0.):
    xprime = compute_xprime(w, F_array)
    xprime_flat = xprime.flatten()
    y_flat = y.flatten()

    cc = pearson_cc(xprime_flat, y_flat)

    l2_penalty = lambda_l2 * jnp.mean((w - 1.0)**2)
    l1_penalty = lambda_l1 * jnp.mean(jnp.abs(w - 1.0))
    
    return cc - l2_penalty - l1_penalty


def optimize_weights(F_array, y, n_steps, step_size, lambda_l1=0., lambda_l2=0.):
    n_timepoints = F_array.shape[0]

    u = jnp.full((n_timepoints,), -jax.numpy.log(n_timepoints - 1))

    print("Initial weights:", jax.nn.sigmoid(u))
    print("Initial CC:", objective(jax.nn.sigmoid(u), F_array, y, lambda_l1, lambda_l2))

    optimizer = adam(step_size)
    params = {"u": u}
    opt_state = optimizer.init(params)

    grad_fn = jit(
        grad(
            lambda params, F_array, y: -objective(
                jax.nn.sigmoid(params["u"]), F_array, y, lambda_l1, lambda_l2
            )
        )
    )

    last_cc = -jnp.inf

    for step in range(n_steps):
        g = grad_fn(params, F_array, y)
        updates, opt_state = optimizer.update(g, opt_state)
        params = apply_updates(params, updates)

        if step % 10 == 0:
            w = jax.nn.sigmoid(params["u"])
            cc = objective(w, F_array, y)
            print(f"Step {step}: Objective = {cc:.6f}")

        if jnp.allclose(cc, last_cc, atol=1e-5):
            print(f"Converged at step {step}")
            break

    w = jax.nn.sigmoid(params["u"])

    return w


if __name__ == "__main__":
    # n_timepoints = 10
    # hkl_shape = (3, 3, 3)

    # key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)

    # good_amplitudes = jax.random.uniform(subkey1, hkl_shape) * 10.0
    # good_phases = jax.random.uniform(subkey2, hkl_shape) * 2 * jnp.pi

    # good_fraction = 0.2
    # n_good = int(n_timepoints * good_fraction)

    # F_good = good_amplitudes[None, :, :, :] * jnp.exp(1j * good_phases[None, :, :, :])
    # F_good = jnp.tile(F_good, (n_good, 1, 1, 1))

    # bad_amplitudes = jax.random.uniform(subkey3, hkl_shape) * 10.0
    # bad_phases = jax.random.uniform(subkey4, (n_timepoints - n_good,) + hkl_shape) * 2 * jnp.pi
    # F_bad = bad_amplitudes[None, :, :, :] * jnp.exp(1j * bad_phases)

    # F_array = jnp.concatenate([F_good, F_bad], axis=0)

    # y_true = jnp.abs(good_amplitudes) ** 2
    # noise = jax.random.normal(subkey5, hkl_shape) * 0.5
    # y = y_true + noise

    gt_dataset = rs.read_mtz(
        "diffUSE_CC_opt_test/sqrtIdiffuse_ground_truth.mtz"
    ).expand_to_p1()
    nas = gt_dataset.isna()
    # gt_dataset.compute_dHKL(inplace=True)
    # gt_dataset = gt_dataset[gt_dataset.dHKL > 2.0]
    gt_dataset = gt_dataset[~nas.sqrtIdiff]
    print("Ground truth:\n", gt_dataset.head())
    y = gt_dataset.sqrtIdiff.to_numpy() ** 2
    print("Ground truth obs shape: ", y.shape)

    datasets = []

    mtzs = glob.glob("diffUSE_CC_opt_test/*.mtz")
    for mtz in mtzs:
        if mtz.startswith("diffUSE_CC_opt_test/sqrtIdiffuse"):
            continue
        print("Reading:", mtz.split("/")[-1])
        dataset = (
            rs.read_mtz(mtz)
            .expand_to_p1()[~nas.sqrtIdiff]
            .to_structurefactor("FMODEL", "PHIFMODEL")
            .to_numpy()
        )
        datasets.append(dataset)

    F_array = jnp.stack(datasets, axis=0)

    print("F_array shape:", F_array.shape)

    weights = optimize_weights(F_array, y, n_steps=500, step_size=0.05, lambda_l1=0.1, lambda_l2=0.)

    print("Datasets:", [mtz.split("/")[-1] for mtz in mtzs if "sqrtIdiffuse" not in mtz])
    print("\nFinal weights:", weights)

    xprime = compute_xprime(weights, F_array)
    print("Final CC:", pearson_cc(xprime, y))
