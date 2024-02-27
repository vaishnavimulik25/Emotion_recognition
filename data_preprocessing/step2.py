import pywt

# Function to perform Discrete Wavelet Transform (DWT)
def dwt_transform(audio, wavelet='db1', level=5):
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    return coeffs

# Function to perform Continuous Wavelet Transform (CWT)
def cwt_transform(audio, wavelet='morl'):
    coefficients, frequencies = pywt.cwt(audio, np.arange(1, 128), wavelet)
    return coefficients

# Function to perform Wavelet Packet Transform (WPT)
def wpt_transform(audio, wavelet='db1', level=5):
    wp = pywt.WaveletPacket(data=audio, wavelet=wavelet, mode='symmetric', maxlevel=level)
    return wp

