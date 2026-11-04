"""
Fenêtre de bureau (Tkinter) : DESSINE un chiffre ou OUVRE une image,
et le modèle K-NN (knn_chiffres.py) donne le chiffre.

Lancer avec :   py -3.12 interface_dessin.py
Tkinter est déjà inclus avec Python sous Windows : rien de plus à installer.
"""

import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from knn_chiffres import charger_modele, image_vers_vecteur, predire

TAILLE = 280          # taille de la zone de dessin (pixels)
EPAISSEUR = 22        # épaisseur du crayon
BLEU = "#1f4fd8"


def couleur_case(v):
    """Valeur 0..16 -> gris (16 = encre noire), au format #rrggbb."""
    g = int(255 - 255 * v / 16)
    return "#%02x%02x%02x" % (g, g, g)


def dessiner_8x8(canvas, vec, taille):
    """Affiche une image 8x8 (64 valeurs) sur un Canvas Tkinter."""
    canvas.delete("all")
    c = taille / 8
    for i in range(8):
        for j in range(8):
            canvas.create_rectangle(j * c, i * c, (j + 1) * c, (i + 1) * c,
                                    fill=couleur_case(vec[i * 8 + j]), outline="")


class Application:
    def __init__(self, racine):
        self.racine = racine
        racine.title("Lecteur de chiffres (K-NN)")
        racine.configure(padx=16, pady=16)

        self.k = 5
        self.cache_modeles = {}
        self.modele, self.X_train, self.X_test, self.y_train, self.y_test = self.modele_pour(self.k)

        self.vec = None        # les 64 valeurs de l'image courante
        self.vrai = None       # vrai chiffre (seulement pour les exemples du jeu de test)
        self.mode = "exemple"  # "exemple", "image" ou "dessin"
        self.dernier = None    # dernier point du crayon
        self.dernier_calcul = 0.0
        self.photo = None      # référence à garder pour l'image affichée

        self.construire()
        self.curseur_k.set(self.k)
        self.changer_k(self.k)     # affiche la précision du modèle
        self.montrer_exemple(0)    # 1re image de test de l'atelier

    # ------------------------------------------------------------------ modèle
    def modele_pour(self, k):
        if k not in self.cache_modeles:
            modele, (X_train, X_test, y_train, y_test) = charger_modele(k)
            self.cache_modeles[k] = (modele, X_train, X_test, y_train, y_test)
        return self.cache_modeles[k]

    # ------------------------------------------------------------- construction
    def construire(self):
        gauche = tk.Frame(self.racine)
        droite = tk.Frame(self.racine)
        gauche.grid(row=0, column=0, sticky="n", padx=(0, 24))
        droite.grid(row=0, column=1, sticky="n")

        # --- colonne de gauche : entrée
        tk.Label(gauche, text="1 · TON CHIFFRE", font=("Segoe UI", 9), fg="#5b6878").pack(anchor="w")
        self.zone = tk.Canvas(gauche, width=TAILLE, height=TAILLE, bg="white",
                              highlightthickness=1, highlightbackground="#cfd6df", cursor="crosshair")
        self.zone.pack(pady=8)
        self.zone.bind("<ButtonPress-1>", self.crayon_appui)
        self.zone.bind("<B1-Motion>", self.crayon_bouge)
        self.zone.bind("<ButtonRelease-1>", self.crayon_relache)

        boutons = tk.Frame(gauche)
        boutons.pack(fill="x")
        tk.Button(boutons, text="Effacer", command=self.effacer).pack(side="left", padx=(0, 6))
        tk.Button(boutons, text="Ouvrir une image…", command=self.ouvrir_image).pack(side="left", padx=(0, 6))
        tk.Button(boutons, text="Exemple du jeu de test", command=self.exemple_aleatoire,
                  bg=BLEU, fg="white").pack(side="left")
        tk.Label(gauche, text="Dessine à la souris, ou ouvre un fichier image.",
                 fg="#5b6878", font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 12))

        self.precision = tk.Label(gauche, text="", fg="#5b6878", font=("Segoe UI", 9))
        self.curseur_k = tk.Scale(gauche, from_=1, to=15, orient="horizontal", length=TAILLE,
                                  label="Nombre de voisins (K)", command=self.changer_k)
        self.curseur_k.pack(anchor="w")
        self.precision.pack(anchor="w", pady=(6, 0))

        # --- colonne de droite : résultat
        tk.Label(droite, text="2 · RÉPONSE DU MODÈLE", font=("Segoe UI", 9), fg="#5b6878").pack(anchor="w")
        haut = tk.Frame(droite)
        haut.pack(anchor="w", pady=8)
        bloc = tk.Frame(haut)
        bloc.pack(side="left", padx=(0, 24))
        self.gros = tk.Label(bloc, text="–", font=("Segoe UI", 96, "bold"), width=2)
        self.gros.pack()
        self.statut = tk.Label(bloc, text="", font=("Segoe UI", 10), fg="#5b6878")
        self.statut.pack()
        vue = tk.Frame(haut)
        vue.pack(side="left")
        self.vue8 = tk.Canvas(vue, width=150, height=150, highlightthickness=1, highlightbackground="#cfd6df")
        self.vue8.pack()
        tk.Label(vue, text="Ce que voit le modèle (8 × 8)", fg="#5b6878", font=("Segoe UI", 8)).pack()

        tk.Label(droite, text="VOTES DES VOISINS", font=("Segoe UI", 9), fg="#5b6878").pack(anchor="w", pady=(8, 2))
        self.votes = tk.Canvas(droite, width=360, height=10 * 18, highlightthickness=0)
        self.votes.pack(anchor="w")

        tk.Label(droite, text="LES VOISINS LES PLUS PROCHES", font=("Segoe UI", 9),
                 fg="#5b6878").pack(anchor="w", pady=(8, 2))
        self.voisins = tk.Frame(droite)
        self.voisins.pack(anchor="w")

    # ------------------------------------------------------------------ entrées
    def effacer(self):
        """Vide la zone de dessin et se prépare à dessiner."""
        self.zone.delete("all")
        self.image_pil = Image.new("L", (TAILLE, TAILLE), 255)   # copie du dessin, pour le modèle
        self.crayon = ImageDraw.Draw(self.image_pil)
        self.mode = "dessin"
        self.vec, self.vrai = None, None
        self.afficher_resultat()

    def crayon_appui(self, e):
        if self.mode != "dessin":
            self.effacer()
        self.dernier = (e.x, e.y)
        self.trait(e.x, e.y, e.x, e.y)

    def crayon_bouge(self, e):
        if self.dernier is None:
            return
        self.trait(self.dernier[0], self.dernier[1], e.x, e.y)
        self.dernier = (e.x, e.y)
        if time.time() - self.dernier_calcul > 0.15:      # prédiction en direct, sans ralentir
            self.lire_dessin()

    def crayon_relache(self, e):
        self.dernier = None
        self.lire_dessin()

    def trait(self, x0, y0, x1, y1):
        """Dessine un trait à l'écran ET dans la copie PIL envoyée au modèle."""
        r = EPAISSEUR / 2
        self.zone.create_line(x0, y0, x1, y1, width=EPAISSEUR, fill="black", capstyle=tk.ROUND)
        self.zone.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="black", outline="black")
        self.crayon.line([x0, y0, x1, y1], fill=0, width=EPAISSEUR)
        self.crayon.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=0)

    def lire_dessin(self):
        self.dernier_calcul = time.time()
        self.vec = image_vers_vecteur(self.image_pil)
        self.vrai = None
        self.afficher_resultat()

    def ouvrir_image(self):
        chemin = filedialog.askopenfilename(
            title="Choisir une image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Tous", "*.*")])
        if not chemin:
            return
        image = Image.open(chemin)
        # aperçu à l'écran (l'original, non modifié, est envoyé au modèle)
        apercu = image.convert("RGB")
        apercu.thumbnail((TAILLE, TAILLE))
        self.photo = ImageTk.PhotoImage(apercu)
        self.zone.delete("all")
        self.zone.create_image(TAILLE // 2, TAILLE // 2, image=self.photo)
        self.mode = "image"
        self.vec, self.vrai = image_vers_vecteur(image), None
        self.afficher_resultat()

    def montrer_exemple(self, i):
        vec = self.X_test[i]
        self.zone.delete("all")
        c = TAILLE / 8
        for a in range(8):
            for b in range(8):
                self.zone.create_rectangle(b * c, a * c, (b + 1) * c, (a + 1) * c,
                                           fill=couleur_case(vec[a * 8 + b]), outline="")
        self.mode = "exemple"
        self.vec, self.vrai = vec, int(self.y_test[i])
        self.afficher_resultat()

    def exemple_aleatoire(self):
        self.montrer_exemple(int(np.random.randint(len(self.y_test))))

    def changer_k(self, valeur):
        self.k = int(valeur)
        self.modele, self.X_train, self.X_test, self.y_train, self.y_test = self.modele_pour(self.k)
        self.precision.config(text="Précision sur le jeu de test : %.2f %%" %
                              (100 * self.modele.score(self.X_test, self.y_test)))
        if hasattr(self, "gros") and self.mode:
            self.afficher_resultat()

    # ----------------------------------------------------------------- résultat
    def afficher_resultat(self):
        self.vue8.delete("all")
        self.votes.delete("all")
        for w in self.voisins.winfo_children():
            w.destroy()

        if self.vec is None:
            self.gros.config(text="–", fg="#5b6878")
            self.statut.config(text="Dessine un chiffre ou ouvre une image.")
            return

        r = predire(self.modele, self.vec, self.y_train)
        chiffre, votes, k = r["chiffre"], r["votes"], self.k
        self.gros.config(text=str(chiffre), fg="black")
        if self.vrai is not None:
            ok = "correct" if self.vrai == chiffre else "erreur"
            self.statut.config(text="Vrai chiffre : %d · %s" % (self.vrai, ok))
        else:
            self.statut.config(text="%d voisin(s) sur %d votent « %d »" % (votes[chiffre], k, chiffre))

        dessiner_8x8(self.vue8, self.vec, 150)

        for d in range(10):
            y = d * 18
            self.votes.create_text(8, y + 9, text=str(d), anchor="center", font=("Consolas", 10))
            self.votes.create_rectangle(24, y + 4, 24 + 280, y + 14, fill="#eef1f6", outline="")
            if votes[d]:
                self.votes.create_rectangle(24, y + 4, 24 + 280 * votes[d] / k, y + 14,
                                            fill=BLEU if d == chiffre else "#9aa6b6", outline="")
            self.votes.create_text(340, y + 9, text=str(votes[d]), anchor="e", font=("Consolas", 10))

        for n, (idx, dist, lab) in enumerate(zip(r["indices"], r["distances"], r["etiquettes_voisins"])):
            cadre = tk.Frame(self.voisins)
            cadre.grid(row=n // 5, column=n % 5, padx=4, pady=4)
            mini = tk.Canvas(cadre, width=56, height=56, highlightthickness=1, highlightbackground="#cfd6df")
            mini.pack()
            dessiner_8x8(mini, self.X_train[idx], 56)
            tk.Label(cadre, text="%d · d=%.1f" % (lab, dist), font=("Consolas", 8)).pack()


if __name__ == "__main__":
    racine = tk.Tk()
    Application(racine)
    racine.mainloop()