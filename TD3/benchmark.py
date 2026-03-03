# benchmark.py
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import version1
import version2

if __name__ == "__main__":
    os.makedirs("benchmark_results", exist_ok=True)

    sizes = [10_000, 100_000, 1_000_000]
    chunk_options = [2, 4, 8, 16]
    schedulers = ['threads', 'processes']

    for scheduler in schedulers:
        all_times_v1 = []
        all_times_v2 = []

        for num_chunks in chunk_options:
            times_v1 = []
            times_v2 = []
            for n in sizes:
                start = time.time()
                version1.run_benchmark(n, num_chunks=num_chunks, scheduler=scheduler)
                t1 = time.time() - start
                times_v1.append(t1)

                start = time.time()
                version2.run_benchmark(n, scheduler=scheduler)
                t2 = time.time() - start
                times_v2.append(t2)

            all_times_v1.append(times_v1)
            all_times_v2.append(times_v2)

        # --- Bar graph regroupé ---
        x = np.arange(len(sizes))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10,6))

        for i, num_chunks in enumerate(chunk_options):
            ax.bar(x + (i - 1.5)*width, all_times_v1[i], width, label=f"V1, chunks={num_chunks}", alpha=0.8)
            ax.bar(x + (i - 1.5 + 0.05)*width, all_times_v2[i], width, label=f"V2, chunks={num_chunks}", alpha=0.8)

        ax.set_xlabel("Taille du tableau")
        ax.set_ylabel("Temps d'exécution (s)")
        ax.set_title(f"Performance globale - Scheduler={scheduler}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s//1000}k" for s in sizes])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.5)

        filename = f"benchmark_results/perf_bar_{scheduler}_all.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph regroupé sauvegardé : {filename}")