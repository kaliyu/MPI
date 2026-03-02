import sys
import time
import os
import dask.array as da
from dask.distributed import Client


def main():
    # usage: python run_dask_3.py n m chunk [delay] [n_workers]
    if len(sys.argv) < 3:
        print("Usage: python3 run_dask_3.py <n> <m> [chunk] [delay] [n_workers]")
        sys.exit(1)

    n = int(sys.argv[1])
    m = int(sys.argv[2])
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, n // 4)
    delay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    n_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    print(f"n={n}, m={m}, chunk={chunk}, delay={delay}")

    # always use distributed client for dashboard
    client = Client(n_workers=n_workers, threads_per_worker=1, 
                   processes=True, dashboard_address=":0")
    print(client)
    print("Dashboard link:", client.dashboard_link)

    # create dask arrays for matrix and vectors
    A = da.random.random((n, n), chunks=(chunk, chunk))
    V = da.random.random((n, m), chunks=(chunk, m))

    # compute product A @ V -> shape (n, m)
    P = A.dot(V)

    # sum the columns (all vectors) to get final vector of length n
    S = P.sum(axis=1)

    # visualize graph of the sum (build once)
    S.visualize(filename="daskarray_ex1.png")
    time.sleep(100)
    start = time.perf_counter()
    future = client.compute(S)
    res = future.result()
    end = time.perf_counter()
    print(f"Temps d'exécution : {end-start:.4f} secondes")
    # optionally show a small portion
    print("résultat (début) :", res[:10])

    client.close()


if __name__ == "__main__":
    main()