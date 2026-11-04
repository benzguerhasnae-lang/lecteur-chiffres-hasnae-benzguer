"""Lecteur de chiffres (K-NN) : interface Streamlit avec zone de dessin et dépôt d'image.

Lancer en local :   python -m streamlit run app.py
Publier en ligne :  voir README.md (Streamlit Community Cloud)
"""

import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from knn_chiffres import charger_modele, image_vers_vecteur, predire

st.set_page_config(page_title="Lecteur de chiffres", page_icon="🔢", layout="wide")

# Composant de dessin : petite page HTML/JS (dossier canvas_frontend) qui renvoie le dessin à Python.
canvas_dessin = components.declare_component(
    "canvas_dessin", path=str(Path(__file__).parent / "canvas_frontend"))


@st.cache_resource
def modele_pour(k):
    """Entraîne le modèle une seule fois par valeur de K (mis en cache)."""
    return charger_modele(k)


def en_image(vec, echelle=24):
    """Agrandit une image 8x8 pour l'afficher (chaque case devient un gros carré)."""
    img = np.kron(np.asarray(vec).reshape(8, 8), np.ones((echelle, echelle)))
    return (255 - 255 * img / 16).astype(np.uint8)        # encre sombre sur fond clair


def image_depuis_data_url(data_url):
    """Transforme le texte « data:image/png;base64,... » envoyé par la zone de dessin en image PIL."""
    if not data_url:
        return None
    octets = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(io.BytesIO(octets))


st.title("Lecteur de chiffres (K-NN)")
st.caption("Dessine un chiffre ou dépose une image : un modèle K-NN entraîné sur 1 257 images "
           "du jeu « digits » de scikit-learn devine de quel chiffre il s'agit.")

k = st.sidebar.slider("Nombre de voisins (K)", 1, 15, 5)
modele, (X_train, X_test, y_train, y_test) = modele_pour(k)
st.sidebar.metric("Précision sur le jeu de test", f"{modele.score(X_test, y_test):.2%}")
st.sidebar.caption("Le modèle a appris sur des chiffres manuscrits : les chiffres tapés avec une "
                   "police sont moins bien reconnus.")

if "exemple" not in st.session_state:
    st.session_state.exemple = 0                          # 1re image de test de l'atelier

col_entree, col_sortie = st.columns(2)
vec, vrai = None, None

with col_entree:
    st.subheader("1 · Ton chiffre")
    mode = st.radio("Source", ["Dessiner", "Déposer une image", "Exemple du jeu de test"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Dessiner":
        data_url = canvas_dessin(key="dessin", default=None)
        st.caption("Trace un gros chiffre, centré, puis relâche la souris.")
        image = image_depuis_data_url(data_url)
        vec = image_vers_vecteur(image) if image is not None else None
        if image is None:
            st.info("Dessine un chiffre dans la zone blanche.")

    elif mode == "Déposer une image":
        fichier = st.file_uploader("Dépose une image (png, jpg)", type=["png", "jpg", "jpeg", "bmp"])
        if fichier is not None:
            image = Image.open(fichier)
            st.image(image, caption="Image choisie", width=250)
            vec = image_vers_vecteur(image)

    else:
        if st.button("Autre exemple au hasard"):
            st.session_state.exemple = int(np.random.randint(len(y_test)))
        i = st.session_state.exemple
        vec, vrai = X_test[i], int(y_test[i])
        st.image(en_image(vec), caption=f"Exemple n°{i} du jeu de test", width=250)

with col_sortie:
    st.subheader("2 · Réponse du modèle")
    if vec is None:
        st.write("En attente d'un chiffre…")
    else:
        r = predire(modele, vec, y_train)
        c1, c2 = st.columns(2)
        c1.markdown(f"<div style='font-size:7rem;font-weight:700;line-height:1'>{r['chiffre']}</div>",
                    unsafe_allow_html=True)
        c2.image(en_image(vec), caption="Ce que voit le modèle (8 × 8)", width=170)
        if vrai is not None:
            st.write(f"Vrai chiffre : **{vrai}** → " + ("✅ correct" if vrai == r["chiffre"] else "❌ erreur"))
        else:
            st.write(f"**{r['votes'][r['chiffre']]} voisins sur {k}** votent « {r['chiffre']} »")

        st.markdown("**Votes des voisins**")
        st.bar_chart({str(d): int(r["votes"][d]) for d in range(10)})

        st.markdown("**Les voisins les plus proches**")
        cols = st.columns(min(k, 5))
        for n, (idx, dist, lab) in enumerate(zip(r["indices"], r["distances"], r["etiquettes_voisins"])):
            cols[n % len(cols)].image(en_image(X_train[idx], 8), caption=f"{lab} · d={dist:.1f}")