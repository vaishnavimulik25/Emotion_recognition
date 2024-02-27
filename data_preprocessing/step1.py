import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

#Function to load audio file
def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

#Function to apply framing and windowing
def apply_framing_windowing(audio, sr, frame_size=1024, hop_size=512, window='hann'):
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
    frames = frames.copy()
    frames *= librosa.filters.get_window(window, frame_size)[:, np.newaxis]
    return frames

#function to remove noise and silence
def remove_noise_silence(audio):
    denoised_audio, _ = librosa.effects.trim(audio)
    return denoised_audio

#Funtion to apply fourier transform
def apply_fourier_transform(frames):
    stft = np.abs(librosa.stft(frames))
    return stft

