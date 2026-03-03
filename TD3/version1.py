# version1.py
import numpy as np
from dask import delayed
from functools import reduce

def run_benchmark(n, num_chunks=8, scheduler='threads'):
    """Retourne la variance d'un tableau de taille n avec Dask.delayed"""
    data = np.random.randint(1, 101, size=n)

    @delayed
    def map_stats(chunk):
        s1 = np.sum(chunk)
        s2 = np.sum(chunk**2)
        n = len(chunk)
        return s1, s2, n

    @delayed
    def reduce_stats(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @delayed
    def compute_variance(stats):
        s1, s2, n = stats
        return s2/n - (s1/n)**2

    chunks = np.array_split(data, num_chunks)
    mapped = [map_stats(c) for c in chunks]
    total = reduce(reduce_stats, mapped)
    variance = compute_variance(total)

    return variance.compute(scheduler=scheduler)