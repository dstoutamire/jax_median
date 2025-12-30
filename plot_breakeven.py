"""Generate plots from breakeven analysis results."""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load breakeven results from JSON file."""
    with open("breakeven_results.json", "r") as f:
        return json.load(f)


def plot_breakeven_timing(results):
    """Create timing comparison across all sizes."""
    sizes = results["sizes"]
    methods = results["methods"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort baseline
    sort_times = [methods["sort"][str(s)]["time"] for s in sizes]
    ax.plot(sizes, sort_times, marker='o', linewidth=2.5, markersize=8,
            label='jnp.median (sort)', color='#1f77b4')

    # Weiszfeld variants
    weiszfeld_colors = {'weiszfeld_2': '#ff7f0e', 'weiszfeld_8': '#2ca02c',
                        'weiszfeld_16': '#9467bd', 'weiszfeld_32': '#bcbd22'}
    for name, color in weiszfeld_colors.items():
        if name in methods:
            times = [methods[name][str(s)]["time"] for s in sizes]
            iters = name.split('_')[1]
            ax.plot(sizes, times, marker='^', linewidth=2, markersize=7,
                    label=f'Weiszfeld-{iters}', color=color)

    # Pivot variants
    pivot_colors = {'pivot_6': '#d62728', 'pivot_8': '#8c564b',
                    'pivot_10': '#e377c2', 'pivot_12': '#7f7f7f'}
    for name, color in pivot_colors.items():
        if name in methods:
            times = [methods[name][str(s)]["time"] for s in sizes]
            iters = name.split('_')[1]
            ax.plot(sizes, times, marker='s', linewidth=2, markersize=7,
                    label=f'Pivot-{iters}', color=color, linestyle='--')

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Breakeven Analysis: Timing Comparison\n(batch size=1000, Cauchy distribution, GPU)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('breakeven_timing.png', dpi=150, bbox_inches='tight')
    print("Saved breakeven_timing.png")


def plot_breakeven_accuracy(results):
    """Create accuracy comparison across all sizes."""
    sizes = results["sizes"]
    methods = results["methods"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Weiszfeld variants
    weiszfeld_colors = {'weiszfeld_2': '#ff7f0e', 'weiszfeld_8': '#2ca02c',
                        'weiszfeld_16': '#9467bd', 'weiszfeld_32': '#bcbd22'}
    for name, color in weiszfeld_colors.items():
        if name in methods:
            mae = [methods[name][str(s)]["mae"] for s in sizes]
            iters = name.split('_')[1]
            ax.plot(sizes, mae, marker='^', linewidth=2, markersize=7,
                    label=f'Weiszfeld-{iters}', color=color)

    # Pivot variants
    pivot_colors = {'pivot_6': '#d62728', 'pivot_8': '#8c564b',
                    'pivot_10': '#e377c2', 'pivot_12': '#7f7f7f'}
    for name, color in pivot_colors.items():
        if name in methods:
            mae = [methods[name][str(s)]["mae"] for s in sizes]
            iters = name.split('_')[1]
            ax.plot(sizes, mae, marker='s', linewidth=2, markersize=7,
                    label=f'Pivot-{iters}', color=color, linestyle='--')

    # Add accuracy target lines
    for target, style in [(0.1, ':'), (0.05, '-.'), (0.01, '--')]:
        ax.axhline(y=target, color='gray', linestyle=style, alpha=0.5,
                   label=f'MAE = {target}')

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Mean Absolute Error vs jnp.median', fontsize=12)
    ax.set_title('Breakeven Analysis: Accuracy Comparison\n(batch size=1000, Cauchy distribution, GPU)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('breakeven_accuracy.png', dpi=150, bbox_inches='tight')
    print("Saved breakeven_accuracy.png")


def plot_best_method_heatmap(results):
    """Create a heatmap showing best method at each accuracy/size combination."""
    sizes = results["sizes"]
    methods = results["methods"]
    accuracy_targets = [0.5, 0.1, 0.05, 0.01, 0.005]

    # Build matrix of speedups for best method
    speedup_matrix = np.zeros((len(accuracy_targets), len(sizes)))
    best_methods = []

    for i, target in enumerate(accuracy_targets):
        row_methods = []
        for j, size in enumerate(sizes):
            best_method = "sort"
            best_time = methods["sort"][str(size)]["time"]

            for method, data in methods.items():
                if method == "sort":
                    continue
                if str(size) in data and data[str(size)]["mae"] <= target:
                    if data[str(size)]["time"] < best_time:
                        best_time = data[str(size)]["time"]
                        best_method = method

            sort_time = methods["sort"][str(size)]["time"]
            speedup = sort_time / best_time if best_method != "sort" else 1.0
            speedup_matrix[i, j] = speedup
            row_methods.append(best_method)
        best_methods.append(row_methods)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(speedup_matrix, cmap='YlGn', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_yticks(np.arange(len(accuracy_targets)))
    ax.set_xticklabels(sizes)
    ax.set_yticklabels([f'MAE ≤ {t}' for t in accuracy_targets])

    # Add text annotations
    for i in range(len(accuracy_targets)):
        for j in range(len(sizes)):
            method = best_methods[i][j]
            speedup = speedup_matrix[i, j]
            short_name = method.replace('weiszfeld_', 'W').replace('pivot_', 'P')
            text = f'{short_name}\n{speedup:.1f}x'
            color = 'black' if speedup < 10 else 'white'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Accuracy Target', fontsize=12)
    ax.set_title('Best Method at Each Accuracy/Size Combination\n(W=Weiszfeld, P=Pivot, speedup vs sort)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label='Speedup vs sort')

    plt.tight_layout()
    plt.savefig('breakeven_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved breakeven_heatmap.png")


def plot_pareto_frontier(results):
    """Create a Pareto frontier plot showing speed vs accuracy tradeoff."""
    sizes = results["sizes"]
    methods = results["methods"]

    # Pick representative sizes from actual benchmark data
    representative_sizes = [1024, 4096, 16384]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, size in zip(axes, representative_sizes):
        # Skip if size not in results
        if str(size) not in methods.get("sort", {}):
            ax.text(0.5, 0.5, f'No data for size {size}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Size = {size}', fontsize=12)
            continue

        # Collect all methods' time and accuracy
        points = []
        for method, data in methods.items():
            if str(size) in data:
                time = data[str(size)]["time"]
                mae = data[str(size)]["mae"]
                points.append((time, mae, method))

        # Sort by time
        points.sort(key=lambda x: x[0])

        # Separate by type
        weiszfeld_points = [(t, m, n) for t, m, n in points if 'weiszfeld' in n]
        pivot_points = [(t, m, n) for t, m, n in points if 'pivot' in n]
        sort_point = [(t, m, n) for t, m, n in points if n == 'sort']

        # Find min MAE > 0 for setting y-axis floor (sort has MAE=0)
        non_zero_maes = [m for t, m, n in points if m > 0]
        min_mae = min(non_zero_maes) if non_zero_maes else 0.001
        y_floor = min_mae / 10  # Put sort point below all others

        # Plot non-sort points
        for pts, marker, color, label in [
            (weiszfeld_points, '^', '#2ca02c', 'Weiszfeld'),
            (pivot_points, 's', '#d62728', 'Pivot')
        ]:
            if pts:
                times = [p[0] for p in pts]
                maes = [p[1] for p in pts]
                ax.scatter(times, maes, marker=marker, s=100, color=color, label=label, alpha=0.8)
                # Add labels
                for t, m, n in pts:
                    iters = n.split('_')[1] if '_' in n else ''
                    ax.annotate(iters, (t, m), textcoords='offset points',
                               xytext=(5, 5), fontsize=8)

        # Plot sort point at y_floor (since MAE=0 can't show on log scale)
        if sort_point:
            t, m, n = sort_point[0]
            ax.scatter([t], [y_floor], marker='o', s=150, color='#1f77b4', label='Sort (exact)', alpha=0.9, zorder=10)
            ax.annotate('exact', (t, y_floor), textcoords='offset points',
                       xytext=(5, 5), fontsize=8, fontweight='bold')

        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'Size = {size}', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Speed vs Accuracy Tradeoff (Pareto Frontier)\nLower-left is better', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('breakeven_pareto.png', dpi=150, bbox_inches='tight')
    print("Saved breakeven_pareto.png")


def main():
    """Generate all breakeven plots."""
    results = load_results()

    plot_breakeven_timing(results)
    plot_breakeven_accuracy(results)
    plot_best_method_heatmap(results)
    plot_pareto_frontier(results)

    print("\nAll breakeven plots generated successfully!")


if __name__ == "__main__":
    main()
