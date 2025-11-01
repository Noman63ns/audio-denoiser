import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===============================
# Streamlit page config
# ===============================
st.set_page_config(page_title="🎧 Audio Denoiser", layout="wide")

st.title("🎧 Audio Denoiser using DeepFilterNet")
st.markdown("Upload an audio file to remove noise using a deep learning model.")

# ===============================
# Load model (cached for speed)
# ===============================
@st.cache_resource
def load_deepfilternet_model():
    model_path = "deepfilternet.h5"  # must exist in root directory
    model = load_model(model_path)
    return model

try:
    model = load_deepfilternet_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ===============================
# Audio helper functions
# ===============================
def load_audio_file(uploaded_file, sr=16000, n_fft=512, hop_length=128):
    y, _ = librosa.load(uploaded_file, sr=sr)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    mag = magnitude.T[np.newaxis, ..., np.newaxis]
    return y, mag, phase

def denoise_audio(y, mag, phase, strength='normal'):
    enhanced_mag = model.predict(mag, verbose=0)[0, ..., 0].T

    if strength == 'normal':
        freq_mask = np.linspace(1, 0.5, enhanced_mag.shape[0])[:, np.newaxis]
    else:
        freq_mask = np.linspace(1, 0.1, enhanced_mag.shape[0])[:, np.newaxis]
        threshold = 0.02 * np.max(np.abs(enhanced_mag))
        enhanced_mag = np.where(enhanced_mag > threshold, enhanced_mag, 0)

    enhanced_mag = np.abs(mag[0, ..., 0].T) * freq_mask
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    y_denoised = librosa.istft(enhanced_stft, hop_length=128)
    return y_denoised

def audio_bytes(y, sr=16000):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    buffer.seek(0)
    return buffer

# ===============================
# Streamlit UI
# ===============================
uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3"])
strength = st.selectbox("Select denoising strength", ["normal", "harsh"])
add_noise = st.checkbox("Add Gaussian noise to test model", value=False)
noise_std = st.slider("Noise standard deviation", 0.0, 0.1, 0.02, 0.005)

if uploaded_file:
    try:
        y, mag, phase = load_audio_file(uploaded_file)
        sr = 16000

        if add_noise:
            y_noisy = y + noise_std * np.random.randn(len(y))
        else:
            y_noisy = y

        with st.spinner("Denoising in progress... ⏳"):
            y_denoised = denoise_audio(y_noisy, mag, phase, strength=strength)

        # ===============================
        # Plot waveforms
        # ===============================
        fig, ax = plt.subplots(3, 1, figsize=(12, 6))
        ax[0].plot(y); ax[0].set_title("Original Audio")
        ax[1].plot(y_noisy); ax[1].set_title("Noisy Audio")
        ax[2].plot(y_denoised); ax[2].set_title(f"Denoised Audio ({strength})")
        plt.tight_layout()
        st.pyplot(fig)

        # ===============================
        # Play audios
        # ===============================
        st.subheader("🎵 Listen to Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Original")
            st.audio(audio_bytes(y), format="audio/wav")

        with col2:
            st.write("Noisy")
            st.audio(audio_bytes(y_noisy), format="audio/wav")

        with col3:
            st.write("Denoised")
            st.audio(audio_bytes(y_denoised), format="audio/wav")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
