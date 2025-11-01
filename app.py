
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import io

st.set_page_config(page_title="Audio Denoiser", layout="wide")

# ===========================
# Build DeepFilterNet model
# ===========================
def build_deepfilternet(freq_bins=257):
    input_layer = layers.Input(shape=(256, freq_bins, 1))
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Conv2D(1, (1,1), activation='relu', padding='same')(x)
    model = models.Model(inputs=input_layer, outputs=output)
    return model

# Create and store model in session state
if "model" not in st.session_state:
    st.session_state.model = build_deepfilternet()
    st.session_state.model.compile(optimizer='adam', loss='mse')

model = st.session_state.model

# ===========================
# Audio processing functions
# ===========================
def load_audio_file(uploaded_file, sr=16000, n_fft=512, hop_length=128):
    y, _ = librosa.load(uploaded_file, sr=sr)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    mag = magnitude.T[np.newaxis, ..., np.newaxis]
    return y, mag, phase

def denoise_audio(y, mag, phase, strength='normal'):
    enhanced_mag = model.predict(mag)[0, ..., 0].T
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

# Helper to convert audio to BytesIO for Streamlit
def audio_bytes(y, sr=16000):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    buffer.seek(0)
    return buffer

# ===========================
# Streamlit UI
# ===========================
st.title("🎧 Audio Denoiser with DeepFilterNet")

uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav","mp3"])
strength = st.selectbox("Select denoising strength", options=["normal","harsh"])
add_noise = st.checkbox("Add Gaussian noise?", value=False)
noise_std = st.slider("Noise standard deviation", min_value=0.0, max_value=0.1, value=0.02, step=0.005)

if uploaded_file:
    y, mag, phase = load_audio_file(uploaded_file)
    sr = 16000
    if add_noise:
        y_noisy = y + noise_std * np.random.randn(len(y))
    else:
        y_noisy = y

    y_denoised = denoise_audio(y_noisy, mag, phase, strength=strength)

    # Plot waveforms
    fig, ax = plt.subplots(3,1, figsize=(12,6))
    ax[0].plot(y); ax[0].set_title("Original Audio")
    ax[1].plot(y_noisy); ax[1].set_title("Noisy Audio")
    ax[2].plot(y_denoised); ax[2].set_title(f"Denoised Audio ({strength})")
    plt.tight_layout()
    st.pyplot(fig)

    # Play audio
    st.subheader("Original Audio")
    st.audio(audio_bytes(y), format='audio/wav')

    st.subheader("Noisy Audio")
    st.audio(audio_bytes(y_noisy), format='audio/wav')

    st.subheader("Denoised Audio")
    st.audio(audio_bytes(y_denoised), format='audio/wav')


