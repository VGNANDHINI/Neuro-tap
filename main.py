import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt

st.title("🧠 NeuroTap: Parkinson’s Detection")

st.markdown("Upload your voice sample below:")

audio_file = st.file_uploader("Upload WAV file", type=["wav"])

if audio_file is not None:
    y, sr = librosa.load(audio_file)
    st.audio(audio_file, format='audio/wav')

    # Feature Extraction Example
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfcc = np.mean(mfcc.T, axis=0)

    st.write("MFCC Features (simplified):", mean_mfcc)

    # Just dummy prediction for now
    risk_score = np.random.choice(["Low", "Moderate", "High"])
    st.success(f"🧪 Predicted Parkinson's Risk: {risk_score}"