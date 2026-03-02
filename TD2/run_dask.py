from dask import delayed
import numpy as np
import sys
import time

def generation(n, m):
    matrice = np.random.rand(n, n)
    vects = np.random.rand(n, m)
    return matrice, vects

@delayed
def prod_mat_vec(mat, v, n):
    return mat @ v

@delayed
def somme(vecteurs, n):
    return np.sum(vecteurs, axis=0)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 run_dask.py <n> <m> <scheduler>")
        sys.exit(1)

    n = int(sys.argv[1])
    m = int(sys.argv[2])
    ordonnanceur = sys.argv[3]

    mat, vecteurs = generation(n, m)
    products = []

    for i in range(m):
        v = vecteurs[:, i]
        products.append(prod_mat_vec(mat, v, n))

    cumul = somme(products, n)
    cumul.visualize(filename="graphe_cumul.png")

    start = time.perf_counter()
    c1 = cumul.compute(scheduler=ordonnanceur, num_workers=4) #modif ici pour tester 
    end = time.perf_counter()
    duration = end - start
    print(f"Temps d'exécution : {duration:.4f} secondes")

if __name__ == "__main__":
    main()
