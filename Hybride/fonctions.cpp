//
// Created by sophie on 18/02/2020.
//
#include <iostream>
#include "fonctions.h"

using namespace std;
void generation_vecteur(int n, int* vecteur, int nb_zero) {
    for (int i=0; i<nb_zero; i++)
        vecteur[i] = 0;
    for (int i=nb_zero; i<n; i++)
        vecteur[i] = rand()%20;

    for (int i=0; i<n; i++)
        cout << vecteur[i] << " ";
    cout << endl;
}

void matrice_vecteur(int n, int* matrice, int* v1, int* v2) {
    int ptr = 0;
    while(v1[ptr]==0)
        ptr++;
    for (int i=0; i<n; i++) {
        v2[i] = 0;
        for (int j = ptr; j < n; j++)
            v2[i] += v1[j] * matrice[i * n + j];
    }
}