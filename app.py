import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import io

st.set_page_config(page_title="Audio Denoiser (PyTorch)", layout="wide")

# ===========================
# Build DeepFilterNet model in PyTorch
# ===========================
class DeepFilterNet(nn.Module):
    def __init__(self, freq_bins=257):
        super(DeepFilterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.conv_out(x))
        return x

# ===========================
# Load model in Streamlit session state
# ===========================
@st.cache_resource
def load_model():
    model = DeepFilterNet()
    model.eval()
    return model

device = torch.device("cpu")
model = load_model().to(device)

# ===========================
# Audio processing functions
# ===========================
def load_audio_file(uploaded_file, sr=16000, n_fft=512, hop_length=128):
    y, _ = librosa.load(uploaded_file, sr=sr)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    mag = torch.tensor(magnitude.T[np.newaxis, ..., np.newaxis], dtype=torch.float32).permute(0,3,1,2)  # [B,C,H,W]
    return y, mag, phase

def denoise_audio(y, mag, phase, strength='normal'):
    with torch.no_grad():
        enhanced_mag = model(mag).cpu().numpy()[0,0].T

    if strength == 'normal':
        freq_mask = np.linspace(1, 0.5, enhanced_mag.shape[0])[:, np.newaxis]
    else:
        freq_mask = np.linspace(1, 0.1, enhanced_mag.shape[0])[:, np.newaxis]
        threshold = 0.02 * np.max(np.abs(enhanced_mag))
        enhanced_mag = np.where(enhanced_mag > threshold, enhanced_mag, 0)

    enhanced_mag = np.abs(mag[0,0].numpy().T) * freq_mask
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    y_denoised = librosa.istft(enhanced_stft, hop_length=128)
    return y_denoised

def audio_bytes(y, sr=16000):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='WAV')
    buffer.seek(0)
    return buffer

# ===========================
# Streamlit UI
# ===========================
st.title("🎧 Audio Denoiser with DeepFilterNet (PyTorch)")

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
