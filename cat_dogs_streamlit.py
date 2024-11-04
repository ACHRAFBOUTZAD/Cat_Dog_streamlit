import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Charger le modèle entraîné
model = tf.keras.models.load_model('cat_dog_model.h5')

# Définir les classes
class_names = ["Chat", "Chien"]

# Préparer l'interface Streamlit
st.title("Predection d'Images : Chats et Chiens")

# Télécharger une image
uploaded_file = st.file_uploader("Choisissez une image de chat ou de chien", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0

    # Afficher l'image téléchargée
    st.image(image, caption='Image téléchargée', width=500)

    # Convertir l'image pour le modèle
    input_image = np.expand_dims(image_array, axis=0)

    # Prédire avec le modèle chargé
    prediction = model.predict(input_image)[0][0]
    classe_predite = "Chien" if prediction > 0.5 else "Chat"
    confiance = prediction if prediction > 0.5 else 1 - prediction

    st.write(f"Classe prédite : {classe_predite} avec une confiance de {confiance * 100:.2f}%")

st.write("Cette application utilise un modèle de classification pour prédire si une image est un chat ou un chien. Vous pouvez télécharger une image pour tester la classification.")
