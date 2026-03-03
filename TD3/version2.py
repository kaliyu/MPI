# version2.py
import dask.array as da

def run_benchmark(n, scheduler='threads'):
    """Retourne la variance d'un tableau de taille n avec Dask Array"""
    # Découpage automatique en chunks de 100_000 max
    x = da.random.randint(1, 101, size=n, chunks=min(n, 100_000))
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    return variance.compute(scheduler=scheduler)