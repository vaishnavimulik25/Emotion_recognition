from step1 import *
from step2 import *
from step3 import *
import soundfile as sf

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
            np.save(stft_output_path, stft)
            
            # Save wavelet transformed audio
            np.save(wavelet_output_path, wavelet_transformed_audio)
            


def main():
    dataset_dir = "./archive/Actor_01"
    output_dir = "./preprocessed_data"
    preprocess_dataset(dataset_dir, output_dir)

if __name__ == "__main__":
    main()
