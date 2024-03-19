from step1 import *
from step2 import *
import soundfile as sf

import os
import numpy as np
import librosa
# Function to preprocess a single audio file
def preprocess_audio_file(file_path):
    audio, sr = load_audio_file(file_path)
    frames = apply_framing_windowing(audio, sr)
    denoised_audio = remove_noise_silence(audio)
    stft = apply_fourier_transform(frames)
    wavelet_transformed_audio = dwt_transform(audio)
    return denoised_audio, stft, wavelet_transformed_audio

def preprocess_dataset(dataset_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(dataset_dir, filename)
            print("Processing:", file_path)
            denoised_audio, stft, wavelet_transformed_audio= preprocess_audio_file(file_path)
            
            # Save preprocessed audio data
            denoised_output_path = os.path.join(output_dir, "denoised_" + filename)
            stft_output_path = os.path.join(output_dir, "stft_" + filename)
            wavelet_output_path = os.path.join(output_dir, "wavelet_" + filename)
            
            # Save denoised audio
            sf.write(denoised_output_path, denoised_audio, 44100)
            
            # Save STFT
            #np.save(stft_output_path, stft)
            
            # Save wavelet transformed audio
            #np.save(wavelet_output_path, wavelet_transformed_audio)

# Function to extract features from audio data
def extract_features(data, sample_rate):
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    
    return np.hstack((zcr, chroma_stft, mfcc, rms, mel))

def preprocess_and_extract_features(dataset_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(dataset_dir):
        if filename.startswith("denoised_") and filename.endswith(".wav"):
            file_path = os.path.join(dataset_dir, filename)
            print("Processing:", file_path)
            
            # Load preprocessed audio
            data, sample_rate = librosa.load(file_path, sr=None)
            
            # Extract features
            features = extract_features(data, sample_rate)
            
            # Save extracted features
            output_path = os.path.join(output_dir, "features_" + filename[len("denoised_"):])
            np.save(output_path, features)
# Function to load extracted features from a directory
def load_features_from_dir(features_dir):
    features = {}
    for filename in os.listdir(features_dir):
        if filename.endswith(".npy"):
            feature_name = filename[:-4]  # Remove the .npy extension
            feature_path = os.path.join(features_dir, filename)
            features[feature_name] = np.load(feature_path)
    return features

def main():
    # print(os.getcwd())
    # return
    dataset_dir = "./RAVDESS/Actor_01"
    output_dir = "./preprocessed_data"
    preprocess_dataset(dataset_dir, output_dir)
    dataset_dir1 = "./preprocessed_data"
    output_dir1 = "./extracted_features/Actor_01"
    preprocess_and_extract_features(dataset_dir1, output_dir1)
    features_dir = "./extracted_features/Actor_01"
    loaded_features = load_features_from_dir(features_dir)
    
    # Print the shape of each loaded feature array
    for feature_name, feature_data in loaded_features.items():
        print(f"Feature: {feature_name}, Shape: {feature_data.shape}")
        #Print the feature data (if desired)
        print(f"Feature Data: {feature_data}")

if __name__ == "__main__":
    main()
