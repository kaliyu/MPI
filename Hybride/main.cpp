#include <iostream>
#include <fstream>
#include <chrono>

#include <mpi.h>
#include <omp.h>

#include "fonctions.h"

using namespace std;

int main(int argc, char **argv) {

  // Pour initialiser l'environnement MPI avec la possibilité d'utiliser des threads (OpenMP)
  int provided; // renvoi le mode d'initialisation effectué
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); // MPI_THREAD_MULITPLE chaque processus MPI peut faire appel à plusieurs threads.

  // Pour connaître son pid et le nombre de processus de l'exécution paralléle (sans les threads)
   int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n = atoi(argv[1]); // taille de la matrice carrée
    int m = atoi(argv[2]); // nombre de vecteurs en entrée

    int root = atoi(argv[3]); // processeur root : référence pour les données

    string name = argv[4]; // le nom du fichier pour que le processus root copie les données initiales et les résultats


    // Petit test pour vérifier qu'on peut avoir plusieurs threads par processus.
#pragma omp parallel num_threads(4)
    {
      int id = omp_get_thread_num();
#pragma omp critical
      cout << "je suis le thread " << id << " pour pid=" << pid << endl;
    }


    // Pour mesurer le temps (géré par le processus root)
    chrono::time_point<chrono::system_clock> debut, fin;

    
    int *matrice = new int[n * n]; // la matrice
    int *vecteurs; // l'ensemble des vecteurs connu uniquement par root et distribué à tous.


    fstream f;
    if (pid == root) {
        f.open(name, std::fstream::out);
        matrice = new int[n * n];
        srand(time(NULL));
        for (int i = 0; i < n * n; i++)
            matrice[i] = rand() % 20;

        f << "Matrice" << endl;
        for (int i = 0; i < n; i++) {
	  for (int j = 0; j < n; j++)
                f << matrice[i * n + j] << " ";
            f << endl;
        }
        f << endl;

        vecteurs = new int[m * n];
        for (int i = 0; i < m; i++) {
            int nb_zero = rand() % (n / 2);
            generation_vecteur(n, vecteurs + i * n, nb_zero);
        }
        f << "Les vecteurs" << endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                f << vecteurs[i * n + j] << " ";
            f << endl;
        }
    }

    // Allocation des buffers locaux
    int vect_per_proc = m / nprocs;  // nombre de vecteurs par processus
    int *local_vecteurs = new int[vect_per_proc * n];
    int *local_resultats = new int[vect_per_proc * n];

    // Broadcast de la matrice à tous les processus
    MPI_Bcast(matrice, n * n, MPI_INT, root, MPI_COMM_WORLD);

    // Scatter des vecteurs
    MPI_Scatter(vecteurs, vect_per_proc * n, MPI_INT,
                local_vecteurs, vect_per_proc * n, MPI_INT,
                root, MPI_COMM_WORLD);

    // Calcul local : matrice × vecteurs avec OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < vect_per_proc; i++) {
        matrice_vecteur(n, matrice, local_vecteurs + i * n, local_resultats + i * n);
    }

    // Dans le temps écoulé on ne s'occupe que de la partie communications et calculs
    // (on oublie la génération des données et l'écriture des résultats sur le fichier de sortie)
    if (pid == root) {
        fin = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = fin - debut;
        cout << "temps en secondes : " << elapsed_seconds.count() << endl;
    }

    int *resultats;
    if (pid == root)
        resultats = new int[m * n];
    
    // A compléter également avec la récupération des résultats
    
    MPI_Gather(local_resultats, vect_per_proc * n, MPI_INT,
               resultats, vect_per_proc * n, MPI_INT,
               root, MPI_COMM_WORLD);
    
    
    if (pid == root) {
      f << "Les vecteurs" << endl;
      for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++)
	  f << resultats[i * n + j] << " ";
	f << endl;
      }
      f.close();
    }
    

    MPI_Finalize();
    return 0;
}
