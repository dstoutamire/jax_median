"""Benchmark JAX median estimation methods."""

import json
import time
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.vmap, in_axes=0)
def smoothed_weiszfeld(x, nu=1e-6, iters=10):
    """Pillutla et al. smoothed Weiszfeld algorithm for robust median estimation."""
    mu = jnp.mean(x)
    for _ in range(iters):
        diff = x - mu
        weights = 1.0 / jnp.sqrt(diff**2 + nu**2)
        mu = jnp.sum(weights * x) / jnp.sum(weights)
    return mu


def make_weiszfeld_fn(iters):
    """Create a Weiszfeld function with specific iteration count."""
    @partial(jax.vmap, in_axes=0)
    def weiszfeld(x, nu=1e-6):
        mu = jnp.mean(x)
        for _ in range(iters):
            diff = x - mu
            weights = 1.0 / jnp.sqrt(diff**2 + nu**2)
            mu = jnp.sum(weights * x) / jnp.sum(weights)
        return mu
    return weiszfeld


def make_nan_weiszfeld_fn(iters):
    """Create a NaN-aware Weiszfeld function with specific iteration count."""
    @partial(jax.vmap, in_axes=0)
    def nan_weiszfeld(x, nu=1e-6):
        # Compute NaN mask once, replace NaN with 0 for safe arithmetic
        nan_mask = jnp.isnan(x)
        x_safe = jnp.where(nan_mask, 0.0, x)
        # Use nanmean for initial estimate
        mu = jnp.nanmean(x)
        for _ in range(iters):
            diff = x_safe - mu
            raw_weights = 1.0 / jnp.sqrt(diff**2 + nu**2)
            weights = jnp.where(nan_mask, 0.0, raw_weights)
            mu = jnp.sum(weights * x_safe) / jnp.sum(weights)
        return mu
    return nan_weiszfeld


def make_pivot_median_fn(iters):
    """Create a pivot-based O(n) median using masks and cumsum.

    This is a GPU-friendly quickselect variant that doesn't physically
    partition data, but uses masks to narrow the search range.
    Expected to be slow due to data-dependent branching and redistribution.
    """
    @partial(jax.vmap, in_axes=0)
    def pivot_median(x):
        n = x.shape[0]
        target = (n - 1) // 2  # Index of median in sorted array

        # Track active range with masks
        active = jnp.ones(n, dtype=bool)
        low_count = 0  # Count of elements known to be below median

        for _ in range(iters):
            # Compute pivot as mean of active elements
            active_vals = jnp.where(active, x, jnp.nan)
            pivot = jnp.nanmean(active_vals)

            # Count elements in each partition (among active elements)
            less_mask = active & (x < pivot)
            equal_mask = active & (x == pivot)
            greater_mask = active & (x > pivot)

            n_less = jnp.sum(less_mask)
            n_equal = jnp.sum(equal_mask)

            # Determine which partition contains the median
            # target is relative to original array, low_count tracks eliminated lower elements
            adjusted_target = target - low_count

            # If median is in "less" partition
            in_less = adjusted_target < n_less
            # If median is in "equal" partition (found it!)
            in_equal = (adjusted_target >= n_less) & (adjusted_target < n_less + n_equal)
            # If median is in "greater" partition
            in_greater = adjusted_target >= n_less + n_equal

            # Update active mask and low_count based on which partition
            new_active = jnp.where(
                in_less, less_mask,
                jnp.where(in_equal, equal_mask, greater_mask)
            )
            new_low_count = jnp.where(
                in_greater, low_count + n_less + n_equal, low_count
            )

            active = new_active
            low_count = new_low_count

        # Return mean of remaining active elements as estimate
        final_vals = jnp.where(active, x, jnp.nan)
        return jnp.nanmean(final_vals)

    return pivot_median


@partial(jax.vmap, in_axes=0)
def batched_median(x):
    """Batched jnp.median."""
    return jnp.median(x)


@partial(jax.vmap, in_axes=0)
def batched_nanmedian(x):
    """Batched jnp.nanmedian."""
    return jnp.nanmedian(x)


def benchmark_fn(fn, data, num_runs=100):
    """Benchmark a function with JIT warm-up."""
    # JIT compile and warm up
    jitted_fn = jax.jit(fn)
    result = jitted_fn(data)
    result.block_until_ready()

    # Time multiple runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = jitted_fn(data)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return jnp.array(times), result


def run_benchmarks():
    """Run all benchmarks and collect results."""
    sizes = [128, 256, 512, 1024]
    batch_size = 100
    num_runs = 100
    weiszfeld_iters = [2, 4, 8, 16]
    pivot_iters = [8, 16, 32]

    key = jax.random.PRNGKey(42)

    results = {
        "sizes": sizes,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "timing": {},
        "accuracy": {},
        "nan_timing": {},
        "nan_accuracy": {}
    }

    # Create functions for different iteration counts
    weiszfeld_fns = {iters: make_weiszfeld_fn(iters) for iters in weiszfeld_iters}
    nan_weiszfeld_fns = {iters: make_nan_weiszfeld_fn(iters) for iters in weiszfeld_iters}
    pivot_fns = {iters: make_pivot_median_fn(iters) for iters in pivot_iters}

    # ========== CLEAN DATA BENCHMARKS ==========
    print("=" * 50)
    print("CLEAN DATA BENCHMARKS (no NaN)")
    print("=" * 50)

    for size in sizes:
        print(f"\nBenchmarking size {size}...")

        # Generate Cauchy data
        key, subkey = jax.random.split(key)
        data = jax.random.cauchy(subkey, shape=(batch_size, size))

        # Benchmark jnp.median
        times_median, result_median = benchmark_fn(batched_median, data, num_runs)
        results["timing"][f"median_{size}"] = {
            "mean": float(jnp.mean(times_median) * 1000),  # ms
            "std": float(jnp.std(times_median) * 1000)
        }
        print(f"  median: {jnp.mean(times_median)*1000:.4f} ± {jnp.std(times_median)*1000:.4f} ms")

        # Benchmark jnp.nanmedian
        times_nanmedian, result_nanmedian = benchmark_fn(batched_nanmedian, data, num_runs)
        results["timing"][f"nanmedian_{size}"] = {
            "mean": float(jnp.mean(times_nanmedian) * 1000),
            "std": float(jnp.std(times_nanmedian) * 1000)
        }
        print(f"  nanmedian: {jnp.mean(times_nanmedian)*1000:.4f} ± {jnp.std(times_nanmedian)*1000:.4f} ms")

        # Benchmark Weiszfeld variants
        for iters in weiszfeld_iters:
            times_weiszfeld, result_weiszfeld = benchmark_fn(weiszfeld_fns[iters], data, num_runs)
            results["timing"][f"weiszfeld_{iters}_{size}"] = {
                "mean": float(jnp.mean(times_weiszfeld) * 1000),
                "std": float(jnp.std(times_weiszfeld) * 1000)
            }

            # Accuracy: MAE vs jnp.median
            mae = float(jnp.mean(jnp.abs(result_weiszfeld - result_median)))
            results["accuracy"][f"weiszfeld_{iters}_{size}"] = mae

            print(f"  weiszfeld(iters={iters}): {jnp.mean(times_weiszfeld)*1000:.4f} ± {jnp.std(times_weiszfeld)*1000:.4f} ms, MAE={mae:.6f}")

        # Benchmark pivot-based median
        for iters in pivot_iters:
            times_pivot, result_pivot = benchmark_fn(pivot_fns[iters], data, num_runs)
            results["timing"][f"pivot_{iters}_{size}"] = {
                "mean": float(jnp.mean(times_pivot) * 1000),
                "std": float(jnp.std(times_pivot) * 1000)
            }

            # Accuracy: MAE vs jnp.median
            mae = float(jnp.mean(jnp.abs(result_pivot - result_median)))
            results["accuracy"][f"pivot_{iters}_{size}"] = mae

            print(f"  pivot(iters={iters}): {jnp.mean(times_pivot)*1000:.4f} ± {jnp.std(times_pivot)*1000:.4f} ms, MAE={mae:.6f}")

    # ========== NaN DATA BENCHMARKS (50% NaN) ==========
    print("\n" + "=" * 50)
    print("NaN DATA BENCHMARKS (50% NaN values)")
    print("=" * 50)

    for size in sizes:
        print(f"\nBenchmarking size {size} with 50% NaN...")

        # Generate Cauchy data with 50% NaN
        key, subkey1, subkey2 = jax.random.split(key, 3)
        data = jax.random.cauchy(subkey1, shape=(batch_size, size))
        nan_mask = jax.random.uniform(subkey2, shape=(batch_size, size)) < 0.5
        data_nan = jnp.where(nan_mask, jnp.nan, data)

        # Benchmark jnp.nanmedian (ground truth for NaN data)
        times_nanmedian, result_nanmedian = benchmark_fn(batched_nanmedian, data_nan, num_runs)
        results["nan_timing"][f"nanmedian_{size}"] = {
            "mean": float(jnp.mean(times_nanmedian) * 1000),
            "std": float(jnp.std(times_nanmedian) * 1000)
        }
        print(f"  nanmedian: {jnp.mean(times_nanmedian)*1000:.4f} ± {jnp.std(times_nanmedian)*1000:.4f} ms")

        # Benchmark NaN-aware Weiszfeld variants
        for iters in weiszfeld_iters:
            times_weiszfeld, result_weiszfeld = benchmark_fn(nan_weiszfeld_fns[iters], data_nan, num_runs)
            results["nan_timing"][f"nan_weiszfeld_{iters}_{size}"] = {
                "mean": float(jnp.mean(times_weiszfeld) * 1000),
                "std": float(jnp.std(times_weiszfeld) * 1000)
            }

            # Accuracy: MAE vs jnp.nanmedian
            mae = float(jnp.nanmean(jnp.abs(result_weiszfeld - result_nanmedian)))
            results["nan_accuracy"][f"nan_weiszfeld_{iters}_{size}"] = mae

            print(f"  nan_weiszfeld(iters={iters}): {jnp.mean(times_weiszfeld)*1000:.4f} ± {jnp.std(times_weiszfeld)*1000:.4f} ms, MAE={mae:.6f}")

    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to benchmark_results.json")
    return results


if __name__ == "__main__":
    results = run_benchmarks()
