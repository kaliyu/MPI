"""
benchmark.py
─────────────────────────────────────────────────────────────────────────────
Axe X  : combinaison  n_workers × threads_per_worker  (ex: "4w×2t")
Axe Y  : temps d'exécution moyen (s)
Courbes: une par taille de chunk (10k / 25k / 50k / 100k)
Figures: 1 PNG par scheduler dans benchmark_results/
─────────────────────────────────────────────────────────────────────────────
"""

import time
import os
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dask.array as da
from dask import delayed
from dask.distributed import Client, LocalCluster

# ── Paramètres ────────────────────────────────────────────────────────────────
N           = 1_000_000
REPEATS     = 15
CHUNK_SIZES = [10_000, 25_000, 50_000, 100_000]
SCHEDULERS  = ['threads', 'processes']
OUT_DIR     = "benchmark_results"

# Combinaisons (n_workers, threads_per_worker) à tester
WORKER_THREAD_COMBOS = [
    (1, 16),
    (2, 8),
    (4, 4),
    (8, 2),
    (16, 1),
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Versions ──────────────────────────────────────────────────────────────────

def v1_run(n, num_chunks, scheduler):
    """Version 1 : Dask delayed + map/reduce manuel."""
    data = np.random.randint(1, 101, size=n)

    @delayed
    def map_stats(chunk):
        return np.sum(chunk), np.sum(chunk ** 2), len(chunk)

    @delayed
    def reduce_stats(a, b):
        return a[0] + b[0], a[1] + b[1], a[2] + b[2]

    @delayed
    def compute_variance(stats):
        s1, s2, n_ = stats
        return s2 / n_ - (s1 / n_) ** 2

    chunks = np.array_split(data, num_chunks)
    mapped = [map_stats(c) for c in chunks]
    total  = reduce(reduce_stats, mapped)
    return compute_variance(total).compute(scheduler=scheduler)


def v2_run(n, chunk_size, scheduler):
    """Version 2 : Dask Array natif."""
    x        = da.random.randint(1, 101, size=n, chunks=chunk_size)
    mean     = x.mean()
    variance = ((x - mean) ** 2).mean()
    return variance.compute(scheduler=scheduler)


# ── Mesure ────────────────────────────────────────────────────────────────────

def measure(fn, repeats=REPEATS, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(**kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ── Benchmark principal ───────────────────────────────────────────────────────

def run_benchmark():
    os.makedirs(OUT_DIR, exist_ok=True)

    for scheduler in SCHEDULERS:
        print(f"\n{'═' * 60}")
        print(f"  Scheduler : {scheduler}")
        print(f"{'═' * 60}")

        # results[version][chunk_size] = liste de temps (un par combo)
        results = {
            'v1': {c: [] for c in CHUNK_SIZES},
            'v2': {c: [] for c in CHUNK_SIZES},
        }
        x_labels = []   # "Nw×Mt"
        x_totals = []   # parallélisme total N*M (pour annotation)

        for n_workers, threads_per_worker in WORKER_THREAD_COMBOS:
            label = f"{n_workers}w×{threads_per_worker}t"
            total = n_workers * threads_per_worker
            x_labels.append(label)
            x_totals.append(total)

            print(f"\n  {label}  (parallélisme total = {total})")

            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=(scheduler == 'processes'),
                silence_logs=40,
            )
            client = Client(cluster)

            try:
                for chunk_size in CHUNK_SIZES:
                    num_chunks = max(1, N // chunk_size)

                    t1 = measure(v1_run, n=N, num_chunks=num_chunks, scheduler=scheduler)
                    results['v1'][chunk_size].append(t1)

                    t2 = measure(v2_run, n=N, chunk_size=chunk_size, scheduler=scheduler)
                    results['v2'][chunk_size].append(t2)

                    print(f"    chunk={chunk_size // 1000:3d}k → V1={t1:.3f}s  V2={t2:.3f}s")

            finally:
                client.close()
                cluster.close()

        # ── Tracé ─────────────────────────────────────────────────────────────
        colors = cm.tab10(np.linspace(0, 0.6, len(CHUNK_SIZES)))
        x_idx  = np.arange(len(x_labels))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        fig.suptitle(
            f"Variance Dask — scheduler={scheduler}  |  n={N:,}  |  moyenne sur {REPEATS} runs",
            fontsize=13, fontweight='bold', y=1.02,
        )

        for ax, vkey, vtitle in [
            (axes[0], 'v1', 'Version 1  (delayed / map-reduce)'),
            (axes[1], 'v2', 'Version 2  (dask array natif)'),
        ]:
            for color, chunk_size in zip(colors, CHUNK_SIZES):
                ax.plot(
                    x_idx,
                    results[vkey][chunk_size],
                    marker='o',
                    linewidth=2.2,
                    markersize=7,
                    color=color,
                    label=f"chunk = {chunk_size // 1000}k",
                )

            ax.set_title(vtitle, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel("Combinaison  workers × threads/worker", fontsize=11)
            ax.set_ylabel("Temps moyen (s)", fontsize=11)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=9)
            ax.legend(title="Taille chunk", fontsize=9, title_fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.45)

            # Annotation "parallélisme total ×N" sous chaque tick
            y_bot = ax.get_ylim()[0]
            for i, total in enumerate(x_totals):
                ax.annotate(
                    f"×{total}",
                    xy=(i, y_bot),
                    xytext=(0, -30),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=7, color='grey',
                )

        plt.tight_layout()
        path = f"{OUT_DIR}/perf_curves_{scheduler}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Graphique sauvegardé : {path}")


if __name__ == "__main__":
    run_benchmark()
    print("\nBenchmark terminé.")