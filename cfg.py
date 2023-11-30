import numpy as np
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
DATA_LEN = 2620 # number of data points
MIN_SIG_LEN = 2  # only use snippets longer than 2 seconds
fs = 16000  # sampling rate
sig_len = 2048  # length of snippet used for tdoa estimation
batch_size = 1 # Batch size for DataLoader



# fetch audio snippets within the range of [0, 2] seconds during training
lower_bound = 0
upper_bound = fs * MIN_SIG_LEN

# room dimensions in meters
dx = 3.0
dy = 2.5
dz = 2.5
room_dim = [dx, dy, dz]
xyz_min = [0.0, 0.0, 0.0]
xyz_max = room_dim

# room properties
t60 = 0.6 # seconds
snr = 5 # dB
c = 343  # speed of sound in m/s

# room dimensions in meters
source_locs = np.random.uniform(low=xyz_min, high=xyz_max, size=(DATA_LEN, 3))


# microphone congifuration
mic_config = 'binaural' # 'binaural', 'triaural', 'tetraural', 'square', 'circular'

# Microphones configurations (microphone locations in meters)
# ------ binarual
binaural_mic_locs = np.array([[1.40, 1.25, 1.80], [1.60, 1.25, 1.80],]).T
binaural_mic_dir =[
            CardioidFamily(orientation=DirectionVector(azimuth=0, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
            CardioidFamily(orientation=DirectionVector(azimuth=180, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),    
            ]
# -------triaural
triaural_mic_locs = np.array([[0.5, 2.25, 1.25], [2.5, 2.25, 1.80], [1.5, 0.25, 1.80]]).T
triaural_mic_dir = [
                CardioidFamily(orientation=DirectionVector(azimuth=315, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
                CardioidFamily(orientation=DirectionVector(azimuth=225, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
                CardioidFamily(orientation=DirectionVector(azimuth=90, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
                ]

# --------tetraural
tetraural_mic_locs = np.array([[0.0, 0.0, 1.8], [3.0, 0.0, 1.8], [0.0, 2.5, 1.8], [3.0, 2.5, 1.8]]).T

# -------- square
square_mic_locs = np.array([[1.4, 1.15, 1.8], [1.6, 1.15, 1.8], [1.4, 1.35, 1.8], [1.6, 1.35, 1.8]]).T

# -------- circular
circular_mic_locs = np.array([[1.6, 1.5, 1.25], [1.5707106781186548, 1.5707106781186548, 1.25], 
    [1.5, 1.6, 1.25], [1.4292893218813452, 1.5707106781186548, 1.25], 
    [1.4, 1.5, 1.25], [1.4292893218813452, 1.4292893218813452, 1.25], 
    [1.5, 1.4, 1.25], [1.5707106781186548, 1.4292893218813452, 1.25]
    ]).T


# DOA Parameters
 # algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS']
algo_names = ['SRP', 'MUSIC', 'TOPS','NormMUSIC', 'WAVES']
nfft = 256  # FFT size
freq_range = [300, 16000] # frequency range for DOA estimation

if __name__ == "__main__":
    print("source_locs.shape: ", source_locs.shape)
    print("source_locs: ", source_locs)
    
