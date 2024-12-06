import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import time
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import io
import base64
from pathlib import Path

# Configuration de la page Streamlit
st.set_page_config(page_title="Détecteur d'Émotions en Direct", layout="wide")

# CSS personnalisé
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stPlotlyChart {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal avec emoji et style
st.title("🎭 Analyseur d'Émotions Intelligent")
st.markdown("---")

# Initialisation des variables dans session state
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
    st.session_state.emotion_history = []
    st.session_state.is_capturing = True
    st.session_state.show_fullscreen = False
    st.session_state.current_frame = None
    st.session_state.emotion_stats = {
        'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0,
        'fear': 0, 'surprise': 0, 'disgust': 0
    }
    st.session_state.video_source = "webcam"
    st.session_state.uploaded_video = None

# Fonction pour créer le rapport PDF
def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, "Rapport d'Analyse des Émotions", 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    
    # Date et durée
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    elapsed_time = int(time.time() - st.session_state.start_time)
    pdf.cell(190, 10, f"Durée de la session: {elapsed_time // 3600:02d}:{(elapsed_time % 3600) // 60:02d}:{elapsed_time % 60:02d}", 0, 1)
    
    # Statistiques globales
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, "Statistiques Globales", 0, 1)
    pdf.set_font('Arial', '', 12)
    
    total_emotions = sum(st.session_state.emotion_stats.values())
    if total_emotions > 0:
        for emotion, count in st.session_state.emotion_stats.items():
            percentage = (count / total_emotions) * 100
            pdf.cell(190, 10, f"{emotion.capitalize()}: {count} ({percentage:.1f}%)", 0, 1)
    
    # Graphique des émotions
    if st.session_state.emotion_history:
        df = pd.DataFrame(st.session_state.emotion_history)
        fig = px.line(df, x=df.index, y='emotion', title="Évolution des émotions")
        fig.write_image("temp_graph.png")
        pdf.ln(10)
        pdf.image("temp_graph.png", x=10, w=190)
        Path("temp_graph.png").unlink()
    
    return pdf.output(dest='S').encode('latin1')

def save_screenshot(frame):
    """Sauvegarde une capture d'écran avec horodatage"""
    if frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_emotion_{timestamp}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return filename
    return None

# Initialisation des variables
WINDOW_SIZE = 30
emotion_counts = {
    'happy': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'sad': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'angry': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'neutral': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'fear': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'surprise': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE),
    'disgust': deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
}

# Sélection de la source vidéo
source_col1, source_col2 = st.columns(2)
with source_col1:
    video_source = st.radio("Choisir la source vidéo:", 
                           ["Webcam", "Vidéo préenregistrée"],
                           key="video_source_radio")

with source_col2:
    if video_source == "Vidéo préenregistrée":
        uploaded_file = st.file_uploader("Choisir une vidéo", 
                                       type=['mp4', 'avi', 'mov'],
                                       key="video_uploader")
        if uploaded_file is not None:
            # Sauvegarder temporairement la vidéo
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.uploaded_video = temp_file
    else:
        st.session_state.uploaded_video = None

# Configuration de la source vidéo
if st.session_state.uploaded_video:
    cap = cv2.VideoCapture(st.session_state.uploaded_video)
else:
    cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Création du layout principal
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Zone de contrôle
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🎥 Pause/Reprise" if st.session_state.is_capturing else "▶️ Reprendre"):
            st.session_state.is_capturing = not st.session_state.is_capturing
    
    with control_col2:
        if st.button("📸 Capture d'écran"):
            if st.session_state.current_frame is not None:
                filename = save_screenshot(st.session_state.current_frame)
                if filename:
                    with open(filename, "rb") as file:
                        btn = st.download_button(
                            label="📥 Télécharger la capture",
                            data=file,
                            file_name=filename,
                            mime="image/png"
                        )
                    # Supprimer le fichier temporaire après le téléchargement
                    Path(filename).unlink(missing_ok=True)
    
    with control_col3:
        if st.button("🔄 Réinitialiser"):
            st.session_state.emotion_stats = {k: 0 for k in st.session_state.emotion_stats}
            st.session_state.emotion_history = []
            st.session_state.start_time = time.time()
    
    with control_col4:
        if st.button("📊 Télécharger Rapport"):
            pdf_bytes = create_pdf_report()
            st.download_button(
                label="📥 Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"rapport_emotions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    # Zone vidéo
    st.subheader("📹 Flux Vidéo")
    video_placeholder = st.empty()

with main_col2:
    st.subheader("📊 Analyse en Temps Réel")
    
    # Métriques en temps réel
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.markdown('<div class="metric-card">⏱️ Temps écoulé</div>', unsafe_allow_html=True)
        time_placeholder = st.empty()
    with metric_cols[1]:
        st.markdown('<div class="metric-card">😊 Émotion dominante</div>', unsafe_allow_html=True)
        emotion_placeholder = st.empty()

    # Graphique des émotions
    graph_placeholder = st.empty()

    # Historique des émotions
    st.subheader("📝 Historique des Émotions")
    history_placeholder = st.empty()

def update_emotion_counts(emotion):
    """Met à jour les compteurs d'émotions"""
    for emotion_type in emotion_counts.keys():
        if emotion_type == emotion:
            emotion_counts[emotion_type].append(1)
            st.session_state.emotion_stats[emotion_type] += 1
        else:
            emotion_counts[emotion_type].append(0)

def create_emotion_graph():
    """Crée un graphique des émotions en temps réel"""
    fig = go.Figure()
    
    colors = {
        'happy': '#2ecc71', 'sad': '#3498db', 
        'angry': '#e74c3c', 'neutral': '#95a5a6',
        'fear': '#9b59b6', 'surprise': '#f1c40f',
        'disgust': '#16a085'
    }
    
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
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

try:
    while True:
        # Mise à jour du temps écoulé
        elapsed_time = int(time.time() - st.session_state.start_time)
        time_placeholder.markdown(f"<h3 style='text-align: center'>{elapsed_time // 3600:02d}:{(elapsed_time % 3600) // 60:02d}:{elapsed_time % 60:02d}</h3>", unsafe_allow_html=True)

        if st.session_state.is_capturing:
            ret, frame = cap.read()
            if not ret:
                if st.session_state.uploaded_video:
                    # Réinitialiser la vidéo si c'est une vidéo préenregistrée
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
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
                    
                    # Mise à jour de l'émotion dominante affichée
                    emotion_placeholder.markdown(f"<h3 style='text-align: center'>{emotion.capitalize()}</h3>", unsafe_allow_html=True)
                    
                    # Ajout à l'historique
                    st.session_state.emotion_history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'emotion': emotion
                    })
                    
                    # Couleurs pour les émotions
                    color = {
                        'happy': (46, 204, 113),
                        'sad': (52, 152, 219),
                        'angry': (231, 76, 60),
                        'neutral': (149, 165, 166),
                        'fear': (155, 89, 182),
                        'surprise': (241, 196, 15),
                        'disgust': (22, 160, 133)
                    }.get(emotion, (255, 255, 255))
                    
                    # Dessin du rectangle et texte
                    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(rgb_frame, f"{emotion.upper()}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Mise à jour des émotions
                    update_emotion_counts(emotion)
                    
                except Exception as e:
                    pass
            
            # Affichage du flux vidéo
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            # Sauvegarder la frame actuelle pour la capture d'écran
            st.session_state.current_frame = rgb_frame.copy()
            
            # Mise à jour du graphique
            current_time = int(time.time() * 1000)
            graph_placeholder.plotly_chart(create_emotion_graph(), use_container_width=True, key=f"emotion_graph_{current_time}")
            
            # Mise à jour de l'historique
            if st.session_state.emotion_history:
                df = pd.DataFrame(st.session_state.emotion_history[-10:])  # Afficher les 10 dernières entrées
                history_placeholder.dataframe(df, use_container_width=True)
            
        time.sleep(0.1)  # Pour réduire l'utilisation CPU

except Exception as e:
    st.error(f"Une erreur s'est produite: {str(e)}")
    
finally:
    cap.release()
    # Nettoyer le fichier vidéo temporaire si nécessaire
    if st.session_state.uploaded_video:
        Path(st.session_state.uploaded_video).unlink(missing_ok=True)
