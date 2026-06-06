# KNN : classification supervisee
# Date : 2024/01/07

from math import sqrt
from random import gauss, shuffle

import matplotlib.pyplot as plt


# Phase 1 : recuperation du jeu de donnees, objet de l'etude.
# Comme on n'en a pas, on va generer des donnees fictives aleatoires.

# 2 classes (ou tag, etiquette, categorie, ...).
# La classe 0 correspond a BLUE, la classe 1 correspond a RED.
classes = ["BLUE", "RED"]

# On genere 2000 donnees etiquetees.
N = 2000

# Format d'une seule donnee : [[x, y], classe 1/0].
# On utilise random.gauss(a, b) pour generer un nombre aleatoire suivant
# une loi gaussienne de moyenne a et d'ecart-type b.
# On en genere autant dans chaque classe.
data = [
    [[gauss(1, 1), gauss(-1, 1)], 1] for i in range(N // 2)
] + [
    [[gauss(-1, 1), gauss(1, 1)], 0] for i in range(N // 2)
]


# Phase 2 : observation, developpement de l'intuition.
# On peut raisonnablement separer le nuage de points en 2 classes,
# meme si certains points rouges sont proches des points bleus, et inversement.
for i in range(len(data)):
    if data[i][1] == 1:
        plt.plot(data[i][0][0], data[i][0][1], "or")  # "or" : rond "o" rouge "r"
    else:
        plt.plot(data[i][0][0], data[i][0][1], "xb")  # "xb" : croix "x" bleue "b"

plt.title("Jeu de donnees fictif pour KNN")
plt.savefig("knn.png")
plt.close()


# Phase 3 : implementation de l'algorithme KNN.

# Distance euclidienne entre 2 points P1 et P2.
def distanceP1_P2(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Calcul de toutes les distances entre un point P et le jeu de donnees.
def distanceP(donnees, p):
    # Retourne une liste de [distance, classe].
    return [[distanceP1_P2(p, pi[0]), pi[1]] for pi in donnees]


# Tri rapide d'une liste contenant [distance, classe], par distance croissante.
def triRapide(L):
    if len(L) <= 1:
        return L

    pivot, L1, L2 = L[0], [], []
    for e in L[1:]:
        if pivot[0] > e[0]:
            L1.append(e)
        else:
            L2.append(e)

    return triRapide(L1) + [pivot] + triRapide(L2)


# Calcul de la classe majoritaire d'un point P en prenant ses k plus proches voisins.
def knn(classes, donnees, p, k):
    classesStats = [0 for _ in range(len(classes))]

    # Tri rapide par distance croissante entre P et les points du jeu de donnees.
    for e in triRapide(distanceP(donnees, p))[:k]:
        # e est au format : [distance, classe].
        classesStats[e[1]] = classesStats[e[1]] + 1

    # Identification de la classe majoritaire.
    res = 0
    for i in range(len(classesStats)):
        if classesStats[i] > classesStats[res]:
            res = i

    return res


# Phase 4 : on melange le jeu de donnees et on le separe
# en donnees d'apprentissage et donnees de test.
_data = data[:]
shuffle(_data)
trainingData = _data[: 9 * len(_data) // 10]
testData = _data[9 * len(_data) // 10 :]


# Phase 5 : evaluation de plusieurs valeurs de k.
# k impair de preference pour mieux cerner la classe majoritaire.
for k in [7, 9, 13, 15, 17]:
    # On initialise une matrice de confusion.
    mtx = [[0 for _ in range(len(classes))] for _ in range(len(classes))]

    # On simule KNN sur chaque point du jeu de donnees test.
    for pi in testData:
        # pi : [[x, y], classe 1/0]
        # classe reelle
        cR = pi[1]

        # classe predite
        cP = knn(classes, trainingData, pi[0], k)

        # remplissage de la matrice de confusion
        mtx[cR][cP] = mtx[cR][cP] + 1

    precision = (mtx[0][0] + mtx[1][1]) / len(testData)
    print("KNN pour k =", k, "sur", len(testData), "tests, matrice de confusion", mtx, "precision", precision)


# Phase 6 : evaluation.
# On choisit le k ayant la meilleure precision.
