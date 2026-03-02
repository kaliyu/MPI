import sys
import time
import os
import dask.array as da


def main():
    # usage: python run_dask_2.py n m scheduler [chunk_size]
    if len(sys.argv) < 4:
        print("Usage: python3 run_dask_2.py <n> <m> <scheduler> [chunk]")
        sys.exit(1)

    n = int(sys.argv[1])
    m = int(sys.argv[2])
    scheduler = sys.argv[3]
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, n // 4)

    print(f"n={n}, m={m}, scheduler={scheduler}, chunk={chunk}")

    # create dask arrays for matrix and vectors
    A = da.random.random((n, n), chunks=(chunk, chunk))
    V = da.random.random((n, m), chunks=(chunk, m))

    # compute product A @ V -> shape (n, m)
    P = A.dot(V)

    # sum the columns (all vectors) to get final vector of length n
    S = P.sum(axis=1)

    # visualize graph of the sum (build once)
    S.visualize(filename="daskarray_ex1.png")

    start = time.perf_counter()
    res = S.compute(scheduler=scheduler)
    end = time.perf_counter()
    print(f"Temps d'exécution : {end-start:.4f} secondes")
    # optionally show a small portion
    print("résultat (début) :", res[:10])


if __name__ == "__main__":
    main()