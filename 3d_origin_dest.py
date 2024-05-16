import numpy as np
import keyboard


# Décomposition en Valeurs Singulières (SVD)
# Pour une matrice M de taille m×n, la décomposition en valeurs singulières est donnée par :
#    M = UΣV^T
# où :
# - U est une matrice orthogonale de taille m×m. Les colonnes de U sont appelées vecteurs singuliers à gauche.
# - Σ est une matrice diagonale de taille m×n avec des valeurs singulières non négatives sur la diagonale.
# - V^T (ou V⊤) est la transposée d'une matrice orthogonale V de taille n×n. Les colonnes de V sont appelées 
# vecteurs singuliers à droite.


# Interprétation des Matrices

# Matrice de Covariance :
# La matrice de covariance H est calculée en multipliant les points d'origine traduits P par les points 
# de destination traduits Q transposés :
#     H = P^T Q
 
# Décomposition en Valeurs Singulières de H :
# La SVD factorise H en trois matrices U, Σ, et V^T. Cette factorisation permet d'extraire les directions 
# principales des données et est utilisée pour calculer la matrice de rotation optimale R.

# Étapes de la SVD

# 1. Calcul de U, Σ, et V^T :
#     La SVD décompose la matrice de covariance H en trois matrices :
#         H = UΣV^T
#     - U : contient les vecteurs singuliers à gauche de H.
#     - Σ : une matrice diagonale avec les valeurs singulières de H.
#     - V^T : contient les vecteurs singuliers à droite de H.
# 2. Calcul de la Matrice de Rotation R :
#     La matrice de rotation R est calculée en multipliant V par U^T (la transposée de U) :
#         R = VU^T


def calculer_barycentre(points):
    """
    Calculer le barycentre d'un ensemble de points.
    """
    return np.mean(points, axis=0)


def algorithme_de_kabsch(Origines, Destinations):
    """
    Algorithme de Kabsch pour trouver la matrice de rotation optimale
    qui aligne les points Origines sur les points Destinations.
    """
    # Calculer la matrice de covariance
    H = np.dot(Origines.T, Destinations)

    # Décomposition en valeurs singulières
    U, S, Vt = np.linalg.svd(H)

    # Calculer la matrice de rotation
    R = np.dot(Vt.T, U.T)

    # Assurer une rotation propre (det(R) doit être 1, pas -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def trouver_matrice_de_transformation(Origines, Destinations):
    """
    Trouver la matrice de transformation qui mappe l'ensemble Origines sur l'ensemble Destinations.
    """
    # Calculer les barycentres de chaque ensemble
    barycentre_Origines = calculer_barycentre(Origines)
    barycentre_Destinations = calculer_barycentre(Destinations)

    # Traduire les points pour amener les barycentres à l'origine
    Origines_translate = Origines - barycentre_Origines
    Destinations_translate = Destinations - barycentre_Destinations

    # Trouver la matrice de rotation optimale en utilisant l'algorithme de Kabsch
    R = algorithme_de_kabsch(Origines_translate, Destinations_translate)

    # Calculer le vecteur de translation
    T = barycentre_Destinations - np.dot(R, barycentre_Origines)

    # Créer la matrice de transformation (3x4)
    matrice_de_transformation = np.hstack((R, T.reshape(-1, 1)))

    return matrice_de_transformation


# Exemple d'utilisation
Origines = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Destinations = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

# Trouver la matrice de transformation 
matrice_de_transformation = trouver_matrice_de_transformation(Origines, Destinations)
print("Matrice de transformation (3x4) :\n", matrice_de_transformation)
 
keyboard.wait('space')

 
# Pour utiliser la matrice de transformation afin d'obetnir les coordonnées 
# Je sais que non nécessaire mais c'est pour tester
def transformer_point(point, matrice_de_transformation):
    """
    Transformer un point en utilisant la matrice de transformation.
    """
    point_transforme = np.dot(
        matrice_de_transformation[:, :3], point) + matrice_de_transformation[:, 3]
    return point_transforme


# EXEMPLE ICI !!!
point_origine = np.array([1., 2., 3.])
print("Point origine :", point_origine)
point_destination = transformer_point(point_origine, matrice_de_transformation)
print("Point transformé :", point_destination)
