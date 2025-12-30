"""Breakeven analysis: find where each algorithm wins at equivalent accuracy."""

import os
# Limit GPU memory to 10%
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def make_weiszfeld_fn(iters):
    @partial(jax.vmap, in_axes=0)
    def weiszfeld(x, nu=1e-6):
        mu = jnp.mean(x)
        for _ in range(iters):
            diff = x - mu
            weights = 1.0 / jnp.sqrt(diff**2 + nu**2)
            mu = jnp.sum(weights * x) / jnp.sum(weights)
        return mu
    return weiszfeld


def make_pivot_fn(iters):
    @partial(jax.vmap, in_axes=0)
    def pivot_median(x):
        n = x.shape[0]
        target = (n - 1) // 2
        active = jnp.ones(n, dtype=bool)
        low_count = 0

        for _ in range(iters):
            active_vals = jnp.where(active, x, jnp.nan)
            pivot = jnp.nanmean(active_vals)

            less_mask = active & (x < pivot)
            equal_mask = active & (x == pivot)
            greater_mask = active & (x > pivot)

            n_less = jnp.sum(less_mask)
            n_equal = jnp.sum(equal_mask)
            adjusted_target = target - low_count

            in_less = adjusted_target < n_less
            in_equal = (adjusted_target >= n_less) & (adjusted_target < n_less + n_equal)
            in_greater = adjusted_target >= n_less + n_equal

            new_active = jnp.where(
                in_less, less_mask,
                jnp.where(in_equal, equal_mask, greater_mask)
            )
            new_low_count = jnp.where(
                in_greater, low_count + n_less + n_equal, low_count
            )

            active = new_active
            low_count = new_low_count

        final_vals = jnp.where(active, x, jnp.nan)
        return jnp.nanmean(final_vals)

    return pivot_median


@partial(jax.vmap, in_axes=0)
def batched_median(x):
    return jnp.median(x)


def benchmark_fn(fn, data, num_runs=50):
    jitted_fn = jax.jit(fn)
    result = jitted_fn(data)
    result.block_until_ready()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = jitted_fn(data)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times) * 1000, np.std(times) * 1000, result


def run_breakeven_analysis():
    """Find breakeven points between sorting, Weiszfeld, and Pivot."""
    # Sizes from 256 to 16384
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    batch_size = 1000
    num_runs = 50

    # Test more iteration counts to find accuracy equivalence
    weiszfeld_iters = [2, 4, 8, 16, 32, 64]
    pivot_iters = [4, 6, 8, 10, 12, 16]

    key = jax.random.PRNGKey(42)

    results = {
        "sizes": sizes,
        "batch_size": batch_size,
        "methods": {}
    }

    # Pre-create all functions
    weiszfeld_fns = {i: make_weiszfeld_fn(i) for i in weiszfeld_iters}
    pivot_fns = {i: make_pivot_fn(i) for i in pivot_iters}

    print("=" * 60)
    print("BREAKEVEN ANALYSIS")
    print("=" * 60)

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Size {size}")
        print("=" * 60)

        key, subkey = jax.random.split(key)
        data = jax.random.cauchy(subkey, shape=(batch_size, size))

        # Baseline: jnp.median (sorting)
        time_sort, std_sort, result_sort = benchmark_fn(batched_median, data, num_runs)
        results["methods"].setdefault("sort", {})[size] = {
            "time": time_sort, "std": std_sort, "mae": 0.0
        }
        print(f"  sort:        {time_sort:7.3f} ± {std_sort:.3f} ms  (exact)")

        # Weiszfeld variants
        for iters in weiszfeld_iters:
            time_w, std_w, result_w = benchmark_fn(weiszfeld_fns[iters], data, num_runs)
            mae = float(jnp.mean(jnp.abs(result_w - result_sort)))
            results["methods"].setdefault(f"weiszfeld_{iters}", {})[size] = {
                "time": time_w, "std": std_w, "mae": mae
            }
            speedup = time_sort / time_w
            print(f"  weiszfeld-{iters:2d}: {time_w:7.3f} ± {std_w:.3f} ms  MAE={mae:.4f}  ({speedup:5.1f}x)")

        # Pivot variants
        for iters in pivot_iters:
            time_p, std_p, result_p = benchmark_fn(pivot_fns[iters], data, num_runs)
            mae = float(jnp.mean(jnp.abs(result_p - result_sort)))
            results["methods"].setdefault(f"pivot_{iters}", {})[size] = {
                "time": time_p, "std": std_p, "mae": mae
            }
            speedup = time_sort / time_p
            print(f"  pivot-{iters:2d}:    {time_p:7.3f} ± {std_p:.3f} ms  MAE={mae:.4f}  ({speedup:5.1f}x)")

    # Save results
    with open("breakeven_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Best method at each accuracy level")
    print("=" * 60)

    # Find best method at different accuracy targets
    accuracy_targets = [0.5, 0.1, 0.05, 0.01, 0.005]

    for target in accuracy_targets:
        print(f"\n  Target MAE ≤ {target}:")
        for size in sizes:
            best_method = "sort"
            best_time = results["methods"]["sort"][size]["time"]

            for method, data in results["methods"].items():
                if method == "sort":
                    continue
                if size in data and data[size]["mae"] <= target:
                    if data[size]["time"] < best_time:
                        best_time = data[size]["time"]
                        best_method = method

            sort_time = results["methods"]["sort"][size]["time"]
            speedup = sort_time / best_time if best_method != "sort" else 1.0
            print(f"    size {size:4d}: {best_method:15s} ({speedup:.1f}x faster than sort)")

    print("\nResults saved to breakeven_results.json")
    return results


if __name__ == "__main__":
    results = run_breakeven_analysis()
