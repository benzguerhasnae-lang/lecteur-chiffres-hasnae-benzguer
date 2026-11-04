"""
Lecteur de chiffres manuscrits avec K-NN (K plus proches voisins).

Ce fichier contient TOUT le « cerveau » du projet :
  1. charger_modele()      -> entraîne le K-NN comme dans l'atelier
  2. image_vers_vecteur()  -> transforme n'importe quelle image en 64 valeurs (8x8)
  3. predire()             -> donne le chiffre prédit, les votes et les voisins

Utilisation en ligne de commande :
    python knn_chiffres.py mon_chiffre.png
    python knn_chiffres.py mon_chiffre.png --k 3
"""

import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------------
# 1) LE MODÈLE : exactement le même que dans l'atelier
# ---------------------------------------------------------------------------
def charger_donnees():
    """Charge le jeu 'digits' et le découpe en apprentissage (70 %) / test (30 %).

    X : tableau de 1797 lignes x 64 colonnes (chaque ligne = une image 8x8 aplatie,
        valeurs de 0 = noir à 16 = blanc/encre).
    y : le vrai chiffre (0 à 9) de chaque image.
    random_state=42 : même découpage que l'atelier, donc mêmes résultats.
    """
    X, y = load_digits(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def charger_modele(k=5):
    """Crée et « entraîne » le K-NN.

    fit() ne calcule rien de compliqué : K-NN se contente de MÉMORISER
    X_train et y_train. Tout le travail se fait plus tard, dans predict().
    """
    X_train, X_test, y_train, y_test = charger_donnees()
    modele = KNeighborsClassifier(n_neighbors=k)
    modele.fit(X_train, y_train)
    return modele, (X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# 2) PRÉTRAITEMENT : une photo/image quelconque -> 64 valeurs comme 'digits'
# ---------------------------------------------------------------------------
def image_vers_vecteur(image, epaisseur_cible=0.11):
    """Transforme une image PIL en vecteur de 64 valeurs (0 à 16), ou None
    si aucun chiffre n'est détecté.

    Pourquoi tant d'étapes ? Le modèle n'a vu QUE des images 8x8 où le chiffre
    est blanc sur fond noir, centré, avec un trait d'épaisseur normale. Une image
    quelconque doit être ramenée à ce même format, sinon les distances n'ont
    aucun sens.
    """
    # a) Niveaux de gris. Si l'image est transparente (PNG), on la pose sur du blanc.
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        fond = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(fond, image)
    gris = image.convert("L")

    # b) On réduit les grandes images (plus rapide, et lisse le bruit).
    gris.thumbnail((200, 200))
    g = np.asarray(gris, dtype=float)
    h, w = g.shape

    # c) « Encre » = écart avec la couleur du fond. Le fond est estimé avec la
    #    médiane des pixels du bord. Ainsi ça marche pour un chiffre noir sur
    #    fond blanc ET pour un chiffre blanc sur fond noir.
    bord = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    fond = np.median(bord)
    encre = np.abs(g - fond)
    if encre.max() < 30:                      # image presque uniforme : rien à lire
        return None

    # d) Masque binaire : un pixel est « encre » s'il est assez foncé/contrasté.
    masque = encre > 0.35 * encre.max()
    ys, xs = np.nonzero(masque)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    grand = max(x1 - x0 + 1, y1 - y0 + 1)     # taille du chiffre en pixels

    # e) Normaliser l'épaisseur du trait. Les chiffres de 'digits' ont un trait
    #    d'environ 11 % de la hauteur du chiffre. On estime l'épaisseur
    #    (2 x surface / périmètre) puis on épaissit (dilatation) ou on
    #    amincit (érosion) pour s'en rapprocher.
    surface = masque.sum()
    p = np.pad(masque, 1)                     # bord de 0 autour, pour compter les côtés exposés
    contour = sum(int((masque & ~p[l:l + h, c:c + w]).sum())
                  for l, c in ((0, 1), (2, 1), (1, 0), (1, 2)))   # haut, bas, gauche, droite
    epaisseur = 2 * surface / max(contour, 1)
    decalage = int(round((epaisseur_cible * grand - epaisseur) / 2))
    decalage = max(-int(epaisseur / 2 - 0.5), min(decalage, int(round(grand * 0.12))))
    if decalage > 0:
        masque = binary_dilation(masque, structure=np.ones((3, 3)), iterations=decalage)
    elif decalage < 0:
        masque = binary_erosion(masque, structure=np.ones((3, 3)), iterations=-decalage)

    # f) Recadrer en CARRÉ centré sur le chiffre, avec un peu de marge
    #    (comme dans 'digits', où le chiffre ne touche pas les bords).
    cote = grand * 1.3 + 2
    cx, cy = (x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2
    sx, sy = cx - cote / 2, cy - cote / 2

    # g) Réduire ce carré en 8x8 par MOYENNE de zone : on prend 8x8 points par
    #    case, et on fait la moyenne (0 = pas d'encre, 1 = tout encre).
    pas = cote / 64
    px = np.floor(sx + (np.arange(64) + 0.5) * pas).astype(int)
    py = np.floor(sy + (np.arange(64) + 0.5) * pas).astype(int)
    valide = ((px >= 0) & (px < w))[None, :] & ((py >= 0) & (py < h))[:, None]
    pts = masque[np.clip(py, 0, h - 1)][:, np.clip(px, 0, w - 1)] * valide
    cases = pts.reshape(8, 8, 8, 8).mean(axis=(1, 3))

    # h) Mettre à l'échelle 0..16 (comme digits : 16 = encre maximale).
    sommet = cases.max()
    if sommet == 0:
        return None
    vecteur = np.clip(np.round(16 * cases / (0.9 * sommet)), 0, 16)
    return vecteur.reshape(-1)               # 8x8 -> 64 valeurs


# ---------------------------------------------------------------------------
# 3) PRÉDICTION : ce que fait K-NN, étape par étape
# ---------------------------------------------------------------------------
def predire(modele, vecteur, y_train):
    """Renvoie un dictionnaire avec le chiffre prédit, les votes et les voisins."""
    x = vecteur.reshape(1, -1)                # scikit-learn attend un tableau 2D
    k = modele.n_neighbors

    # 1) distances (euclidiennes) entre x et TOUTES les images d'entraînement,
    #    puis on garde les K plus petites : ce sont les K plus proches voisins.
    distances, indices = modele.kneighbors(x, n_neighbors=k)
    distances, indices = distances[0], indices[0]

    # 2) on regarde l'étiquette (le vrai chiffre) de chacun de ces voisins.
    etiquettes = y_train[indices]

    # 3) vote : on compte combien de voisins ont chaque chiffre.
    votes = np.bincount(etiquettes, minlength=10)

    # 4) le chiffre qui a le plus de votes gagne (égalité : le plus petit chiffre,
    #    comme scikit-learn).
    chiffre = int(np.argmax(votes))

    # Vérification : doit donner la même chose que modele.predict(x).
    assert chiffre == int(modele.predict(x)[0])
    return {"chiffre": chiffre, "votes": votes, "indices": indices,
            "distances": distances, "etiquettes_voisins": etiquettes}


# ---------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lire un chiffre avec K-NN.")
    parser.add_argument("image", help="chemin d'une image (png, jpg...)")
    parser.add_argument("--k", type=int, default=5, help="nombre de voisins (défaut 5)")
    args = parser.parse_args()

    modele, (X_train, X_test, y_train, y_test) = charger_modele(args.k)
    print(f"Précision sur le jeu de test : {modele.score(X_test, y_test):.4f}")

    vec = image_vers_vecteur(Image.open(args.image))
    if vec is None:
        print("Aucun chiffre détecté dans l'image.")
    else:
        r = predire(modele, vec, y_train)
        print(f"Chiffre prédit : {r['chiffre']}")
        print("Étiquettes des voisins :", r["etiquettes_voisins"].tolist())
        print("Votes par chiffre      :", r["votes"].tolist())