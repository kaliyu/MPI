"""
benchmark_workers_axis.py
─────────────────────────────────────────────────────────────────────────────
chunks  : 100% automatique (chunks='auto' V2, num_chunks=workers×threads V1)
Axe X   : combinaisons workers × threads  (1×16, 2×8, 4×4, 8×2, 16×1)
Axe Y   : temps moyen (s)
Courbes : V1-threads, V1-processes, V2-threads, V2-processes
Figure  : 1 seul PNG à la fin  →  perf_workers_axis.png

CORRECTION : utilise client.compute() pour que le LocalCluster soit
             réellement sollicité (et non bypasse par scheduler='threads')
─────────────────────────────────────────────────────────────────────────────
"""

import time
import os
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from dask import delayed
from dask.distributed import Client, LocalCluster, wait

# ── Paramètres ────────────────────────────────────────────────────────────────
N_FIXED    = 1_000_000
REPEATS    = 3
SCHEDULERS = ['threads', 'processes']
OUT_DIR    = "benchmark_results"

WORKER_THREAD_COMBOS = [
    (1,  16),
    (2,   8),
    (4,   4),
    (8,   2),
    (16,  1),
]
X_LABELS = [f"{w}w×{t}t" for w, t in WORKER_THREAD_COMBOS]
# ──────────────────────────────────────────────────────────────────────────────


# ── Versions ──────────────────────────────────────────────────────────────────

def build_v1(n, n_workers, threads_per_worker):
    """Construit le graphe delayed V1 sans l'exécuter."""
    data       = np.random.randint(1, 101, size=n)
    num_chunks = max(1, n_workers * threads_per_worker)

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
    return compute_variance(total)   # objet delayed, pas encore exécuté


def build_v2(n):
    """Construit le graphe dask array V2 sans l'exécuter."""
    x = da.random.randint(1, 101, size=n, chunks='auto')
    return ((x - x.mean()) ** 2).mean()   # objet dask, pas encore exécuté


# ── Mesure via le cluster ─────────────────────────────────────────────────────

def measure_with_client(build_fn, client, repeats=REPEATS, **build_kwargs):
    """
    Exécute build_fn(**build_kwargs) pour obtenir un graphe Dask,
    puis le soumet au cluster via client.compute() — le LocalCluster
    est donc réellement utilisé.
    """
    times = []
    for _ in range(repeats):
        graph = build_fn(**build_kwargs)
        t0    = time.perf_counter()
        future = client.compute(graph)
        future.result()               # attend la fin sur le cluster
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collecte de TOUS les résultats avant de tracer
    all_results = {}

    for scheduler in SCHEDULERS:
        print(f"\n{'═' * 64}")
        print(f"  Scheduler : {scheduler}  |  n={N_FIXED:,}  |  chunks=auto")
        print(f"{'═' * 64}")

        times_v1 = []
        times_v2 = []

        for (n_workers, threads_per_worker), label in zip(WORKER_THREAD_COMBOS, X_LABELS):
            print(f"\n  {label}  (parallélisme total ×{n_workers * threads_per_worker})")

            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=(scheduler == 'processes'),
                silence_logs=40,
            )
            client = Client(cluster)

            try:
                t1 = measure_with_client(
                    build_v1, client,
                    n=N_FIXED,
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                )
                t2 = measure_with_client(
                    build_v2, client,
                    n=N_FIXED,
                )

                times_v1.append(t1)
                times_v2.append(t2)
                print(f"    → V1={t1:.3f}s  V2={t2:.3f}s")

            finally:
                client.close()
                cluster.close()

        all_results[scheduler] = {'v1': times_v1, 'v2': times_v2}

    # ── Tracé unique une fois TOUS les schedulers terminés ────────────────────
    print("\n  Tous les schedulers terminés — génération du graphique...")

    x_idx  = np.arange(len(X_LABELS))
    styles = {
        ('threads',   'v1'): dict(color='steelblue', linestyle='-',  marker='o'),
        ('threads',   'v2'): dict(color='tomato',    linestyle='-',  marker='s'),
        ('processes', 'v1'): dict(color='steelblue', linestyle='--', marker='o'),
        ('processes', 'v2'): dict(color='tomato',    linestyle='--', marker='s'),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        f"Variance Dask  —  chunks=auto  |  n={N_FIXED:,}  |  moy. {REPEATS} runs\n"
        f"Parallélisme total constant ×16  —  répartition workers/threads variable\n"
        f"Exécution via LocalCluster (client.compute)",
        fontsize=12, fontweight='bold', y=1.03,
    )

    for scheduler in SCHEDULERS:
        for vkey, vlabel in [('v1', 'delayed'), ('v2', 'dask array')]:
            s = styles[(scheduler, vkey)]
            ax.plot(
                x_idx,
                all_results[scheduler][vkey],
                linewidth=2.2, markersize=8,
                label=f"{vlabel}  [{scheduler}]",
                **s,
            )

    ax.set_xlabel("Combinaison workers × threads/worker", fontsize=11)
    ax.set_ylabel("Temps moyen (s)", fontsize=11)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(X_LABELS, fontsize=10)
    ax.legend(fontsize=10, title="version  [scheduler]", title_fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    path = f"{OUT_DIR}/perf_workers_axis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓  Graphique sauvegardé : {path}")


if __name__ == "__main__":
    run()
    print("\nBenchmark terminé. Fichier dans benchmark_results/")