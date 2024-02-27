import numpy as np
import librosa
import random

# Function to resample audio
def resample_audio(audio, target_sr):
    return librosa.resample(audio, orig_sr=len(audio), target_sr=target_sr)

# Function to add Gaussian noise to audio
def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_level * noise
    return augmented_audio

# Function to apply Vocal-Tract Length Perturbation (VLTP)
def vocal_tract_length_perturbation(audio, alpha_range=(0.9, 1.1)):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return librosa.effects.time_stretch(audio, rate=alpha)

# Function to add random white noise
def add_random_white_noise(audio, noise_level=0.005):
    noise = np.random.uniform(-1, 1, len(audio))
    augmented_audio = audio + noise_level * noise
    return augmented_audio

# Main data augmentation function
def augment_data(audio, sr):
    augmented_audios = []
    # Resample
    target_sr = sr + random.randint(-2000, 2000)  # Simulating different sampling frequencies
    augmented_audio = resample_audio(audio, target_sr)
    augmented_audios.append(augmented_audio)
    
    # Add Gaussian noise
    augmented_audio = add_gaussian_noise(audio)
    augmented_audios.append(augmented_audio)
    
    # Apply Vocal-Tract Length Perturbation (VLTP)
    augmented_audio = vocal_tract_length_perturbation(audio)
    augmented_audios.append(augmented_audio)
    
    # Add random white noise
    augmented_audio = add_random_white_noise(audio)
    augmented_audios.append(augmented_audio)
    
    return augmented_audios


