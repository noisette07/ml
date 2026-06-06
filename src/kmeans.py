# Clustering, apprentissage non supervise
# Date : 2024/01/07

from math import sqrt
from random import gauss, shuffle

import matplotlib.pyplot as plt


# Usage possible : compression d'image en 16 couleurs.
# Les couleurs proches sont regroupees, puis remplacees par une couleur moyenne.

# Principe :
# 1. On choisit arbitrairement k centres dans le jeu de donnees.
# 2. Chacun des points du jeu de donnees est associe au barycentre le plus proche.
# 3. On calcule les barycentres de ces k nouveaux groupes.
# 4. On reprend a l'etape 2 jusqu'a convergence des centres.


# Phase 1 : generation de donnees fictives.
N = 1500
data = (
    [[gauss(1, 0.7), gauss(-1, 0.7)] for i in range(N // 3)]
    + [[gauss(-1, 0.7), gauss(1, 0.7)] for i in range(N // 3)]
    + [[gauss(2, 0.7), gauss(2, 0.7)] for i in range(N // 3)]
)


def distanceP1_P2(p1, p2):
    # Distance euclidienne entre 2 points, valable en dimension quelconque.
    d = 0
    for i in range(len(p1)):
        d = d + (p1[i] - p2[i]) ** 2

    return sqrt(d)


def barycentre(pointsListe):
    # Calcul du barycentre d'un ensemble de points.
    dim = len(pointsListe[0])
    coordG = [0 for _ in range(dim)]

    for i in range(dim):
        for e in pointsListe:
            coordG[i] = coordG[i] + e[i]

        coordG[i] = coordG[i] / len(pointsListe)

    return coordG


def choisirCentresAleatoires(pointsListe, k):
    # Tirage aleatoire de k centres parmi un ensemble de points.
    _pointsListe = pointsListe[:]
    shuffle(_pointsListe)

    return _pointsListe[:k]


def calculBarycentresGroupes(groupes, barycentresGroupesOld):
    # Calcul des barycentres de chaque groupe.
    k = len(groupes)
    res = []

    for i in range(k):
        if len(groupes[i]) == 0:
            # Si un groupe devient vide, on garde son ancien barycentre.
            res.append(barycentresGroupesOld[i])
        else:
            res.append(barycentre(groupes[i]))

    return res


def centreLePlusProcheDeP(barycentresGroupes, p):
    # Recherche du centre le plus proche d'un point P.
    k = len(barycentresGroupes)
    distanceMin = float("inf")
    indiceMin = 0

    for i in range(k):
        _d = distanceP1_P2(p, barycentresGroupes[i])
        if _d < distanceMin:
            distanceMin = _d
            indiceMin = i

    return indiceMin


def affecterPointAuGroupeLePlusProche(donnees, barycentresGroupes):
    # Liste de listes, une par groupe, contenant les points associes.
    groupes = [[] for _ in range(len(barycentresGroupes))]

    for p in donnees:
        centre = centreLePlusProcheDeP(barycentresGroupes, p)
        groupes[centre].append(p)

    return groupes


def kmeans(donnees, k):
    barycentresGroupesOld = None
    barycentresGroupesNew = choisirCentresAleatoires(donnees, k)
    maxIterations = 100
    iteration = 0

    while barycentresGroupesOld != barycentresGroupesNew and iteration < maxIterations:
        barycentresGroupesOld = barycentresGroupesNew
        groupes = affecterPointAuGroupeLePlusProche(donnees, barycentresGroupesOld)
        barycentresGroupesNew = calculBarycentresGroupes(groupes, barycentresGroupesOld)
        iteration = iteration + 1

    return groupes


# Recherche de k = 3 groupes.
groupes = kmeans(data, 3)


# Visualisation du resultat.
couleurs = ["or", "xb", "sg"]

for i in range(len(groupes)):
    for p in groupes[i]:
        plt.plot(p[0], p[1], couleurs[i], markersize=2)

plt.title("Clustering K-means")
plt.savefig("clustering.png")
plt.close()

for i in range(len(groupes)):
    print("Groupe", i + 1, ":", len(groupes[i]), "points")
