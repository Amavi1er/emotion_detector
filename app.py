import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import plotly.graph_objects as go
from collections import deque
import time

# Configuration de la page Streamlit
st.set_page_config(page_title="Détecteur d'Émotions en Direct", layout="wide")
st.title("📊 Analyse des Émotions en Temps Réel 🎭")

# Initialisation des variables
WINDOW_SIZE = 30  # Nombre de points dans le graphique
emotion_counts = {
    'happy': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'sad': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'angry': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'neutral': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
}

# Configuration de la webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Création des colonnes pour l'interface
col1, col2 = st.columns(2)

# Placeholder pour le flux vidéo
with col1:
    st.header("Flux Vidéo")
    video_placeholder = st.empty()

# Placeholder pour le graphique
with col2:
    st.header("Analyse des Émotions")
    graph_placeholder = st.empty()

def update_emotion_counts(emotion):
    """Met à jour les compteurs d'émotions"""
    for emotion_type in emotion_counts.keys():
        if emotion_type == emotion:
            emotion_counts[emotion_type].append(1)
        else:
            emotion_counts[emotion_type].append(0)

def create_emotion_graph():
    """Crée un graphique des émotions en temps réel"""
    fig = go.Figure()
    
    colors = {'happy': '#2ecc71', 'sad': '#3498db', 
              'angry': '#e74c3c', 'neutral': '#95a5a6'}
    
    for emotion, counts in emotion_counts.items():
        fig.add_trace(go.Scatter(
            y=list(counts),
            name=emotion.capitalize(),
            line=dict(color=colors[emotion], width=2),
            fill='tozeroy'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Conversion en RGB pour DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection des visages
        faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            try:
                # Analyse des émotions avec DeepFace
                result = DeepFace.analyze(rgb_frame[y:y+h, x:x+w], 
                                        actions=['emotion'],
                                        enforce_detection=False)
                
                # Récupération de l'émotion dominante
                emotion = result[0]['dominant_emotion']
                
                # Dessin du rectangle autour du visage
                color = {
                    'happy': (46, 204, 113),
                    'sad': (52, 152, 219),
                    'angry': (231, 76, 60),
                    'neutral': (149, 165, 166)
                }.get(emotion, (255, 255, 255))
                
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(rgb_frame, emotion.upper(), (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Mise à jour des émotions
                update_emotion_counts(emotion)
                
            except Exception as e:
                pass
        
        # Affichage du flux vidéo
        video_placeholder.image(rgb_frame, channels="RGB")
        
        # Mise à jour du graphique
        graph_placeholder.plotly_chart(create_emotion_graph(), use_container_width=True)
        
        time.sleep(0.1)  # Pour réduire l'utilisation CPU

except Exception as e:
    st.error(f"Une erreur s'est produite: {str(e)}")
    
finally:
    cap.release()
