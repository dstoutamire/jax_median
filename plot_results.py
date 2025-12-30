"""Generate plots from benchmark results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results():
    """Load benchmark results from JSON file."""
    with open("benchmark_results.json", "r") as f:
        return json.load(f)


def plot_cauchy_distribution():
    """Plot the Cauchy distribution used in benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate samples
    np.random.seed(42)
    samples = stats.cauchy.rvs(size=10000)
    samples_clipped = np.clip(samples, -10, 10)

    # Histogram of samples
    ax.hist(samples_clipped, bins=80, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label='10,000 samples (clipped)')

    # Overlay theoretical PDF
    x = np.linspace(-10, 10, 1000)
    pdf = stats.cauchy.pdf(x)
    ax.plot(x, pdf, 'r-', linewidth=2.5, label='Theoretical PDF')

    # Mark the median
    ax.axvline(x=0, color='darkgreen', linestyle='--', linewidth=2, label='Median = 0')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Standard Cauchy Distribution (location=0, scale=1)\nUsed for benchmarking median estimators', fontsize=14)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.4)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cauchy_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved cauchy_distribution.png")


def plot_timing(results):
    """Create timing comparison plot including pivot."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.array(sizes)

    # jnp.median and nanmedian
    median_times = [results["timing"][f"median_{s}"]["mean"] for s in sizes]
    ax.plot(x, median_times, marker='o', label='jnp.median (sort)', linewidth=2, markersize=8, color='#1f77b4')

    # Weiszfeld variants
    for iters, marker, color in [(2, '^', '#ff7f0e'), (8, 'D', '#2ca02c'), (16, 'p', '#9467bd')]:
        times = [results["timing"][f"weiszfeld_{iters}_{s}"]["mean"] for s in sizes]
        ax.plot(x, times, marker=marker, label=f'Weiszfeld ({iters} iters)', linewidth=2, markersize=8, color=color)

    # Pivot variants
    for iters, marker, color in [(8, 's', '#d62728'), (16, 'v', '#8c564b')]:
        times = [results["timing"][f"pivot_{iters}_{s}"]["mean"] for s in sizes]
        ax.plot(x, times, marker=marker, label=f'Pivot ({iters} iters)', linewidth=2, markersize=8, color=color, linestyle='--')

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Median Estimation: Timing Comparison\n(batch size={batch_size}, Cauchy distribution, CPU)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved timing_comparison.png")


def plot_accuracy(results):
    """Create accuracy comparison plot for all methods."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(sizes))
    width = 0.12

    # Weiszfeld variants
    weiszfeld_colors = ['#ff7f0e', '#2ca02c', '#9467bd']
    for i, iters in enumerate([2, 8, 16]):
        mae_values = [results["accuracy"][f"weiszfeld_{iters}_{s}"] for s in sizes]
        ax.bar(x + i * width, mae_values, width, label=f'Weiszfeld ({iters})', color=weiszfeld_colors[i])

    # Pivot variants
    pivot_colors = ['#d62728', '#8c564b']
    for i, iters in enumerate([8, 16]):
        mae_values = [results["accuracy"][f"pivot_{iters}_{s}"] for s in sizes]
        ax.bar(x + (3 + i) * width, mae_values, width, label=f'Pivot ({iters})', color=pivot_colors[i], hatch='//')

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Mean Absolute Error vs jnp.median', fontsize=12)
    ax.set_title(f'Algorithm Accuracy Comparison\n(batch size={batch_size}, Cauchy distribution)', fontsize=14)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved accuracy_comparison.png")


def plot_speedup(results):
    """Create speedup comparison plot."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Weiszfeld
    for iters in [2, 8, 16]:
        speedups = [results["timing"][f"median_{s}"]["mean"] / results["timing"][f"weiszfeld_{iters}_{s}"]["mean"] for s in sizes]
        ax.plot(sizes, speedups, marker='o', label=f'Weiszfeld ({iters} iters)', linewidth=2, markersize=8)

    # Pivot
    for iters in [8, 16]:
        speedups = [results["timing"][f"median_{s}"]["mean"] / results["timing"][f"pivot_{iters}_{s}"]["mean"] for s in sizes]
        ax.plot(sizes, speedups, marker='s', label=f'Pivot ({iters} iters)', linewidth=2, markersize=8, linestyle='--')

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Speedup vs jnp.median', fontsize=12)
    ax.set_title(f'Speedup over jnp.median\n(batch size={batch_size}, Cauchy distribution)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved speedup_comparison.png")


def plot_nan_timing(results):
    """Create timing comparison plot for NaN data."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]

    nanmedian_times = [results["nan_timing"][f"nanmedian_{s}"]["mean"] for s in sizes]
    nanmedian_std = [results["nan_timing"][f"nanmedian_{s}"]["std"] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(sizes)

    ax.errorbar(x, nanmedian_times, yerr=nanmedian_std, marker='s', label='jnp.nanmedian', capsize=3, linewidth=2, markersize=8, color='#1f77b4')

    for iters, marker, color in [(2, '^', '#ff7f0e'), (8, 'D', '#2ca02c'), (16, 'p', '#9467bd')]:
        times = [results["nan_timing"][f"nan_weiszfeld_{iters}_{s}"]["mean"] for s in sizes]
        std = [results["nan_timing"][f"nan_weiszfeld_{iters}_{s}"]["std"] for s in sizes]
        ax.errorbar(x, times, yerr=std, marker=marker, label=f'nan_weiszfeld ({iters} iters)', capsize=3, linewidth=2, markersize=8, color=color)

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'NaN-aware Median Estimation: Timing Comparison\n(batch size={batch_size}, 50% NaN, Cauchy distribution)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nan_timing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved nan_timing_comparison.png")


def plot_nan_accuracy(results):
    """Create accuracy comparison plot for NaN-aware Weiszfeld variants."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]
    iters_list = [2, 4, 8, 16]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sizes))
    width = 0.2

    colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    for i, iters in enumerate(iters_list):
        mae_values = [results["nan_accuracy"][f"nan_weiszfeld_{iters}_{s}"] for s in sizes]
        bars = ax.bar(x + i * width, mae_values, width, label=f'nan_weiszfeld ({iters} iters)', color=colors[i])

        for bar, val in zip(bars, mae_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Mean Absolute Error vs jnp.nanmedian', fontsize=12)
    ax.set_title(f'NaN-aware Weiszfeld Accuracy\n(batch size={batch_size}, 50% NaN, Cauchy distribution)', fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('nan_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved nan_accuracy_comparison.png")


def plot_nan_speedup(results):
    """Create speedup comparison plot for NaN data."""
    sizes = results["sizes"]
    batch_size = results["batch_size"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for iters in [2, 4, 8, 16]:
        speedups = [results["nan_timing"][f"nanmedian_{s}"]["mean"] / results["nan_timing"][f"nan_weiszfeld_{iters}_{s}"]["mean"] for s in sizes]
        ax.plot(sizes, speedups, marker='o', label=f'nan_weiszfeld ({iters} iters)', linewidth=2, markersize=8)

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Speedup vs jnp.nanmedian', fontsize=12)
    ax.set_title(f'NaN-aware Weiszfeld Speedup over jnp.nanmedian\n(batch size={batch_size}, 50% NaN, Cauchy distribution)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('nan_speedup_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved nan_speedup_comparison.png")


def main():
    """Generate all plots."""
    results = load_results()

    # Clean data plots
    plot_cauchy_distribution()
    plot_timing(results)
    plot_accuracy(results)
    plot_speedup(results)

    # NaN data plots
    plot_nan_timing(results)
    plot_nan_accuracy(results)
    plot_nan_speedup(results)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
