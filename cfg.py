import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

# DOA Parameters
# algo_names = ['SRP', 'MUSIC', 'FRIDA', 'TOPS']
algo_names = ['SRP', 'MUSIC','NormMUSIC', 'TOPS', 'WAVES', 'FRIDA']
nfft = 256  # FFT size
freq_range = [300, 16000] # frequency range for DOA estimation


MAX_DATA_LEN = 2620  # number of data points in Librispeeh test-clean
N_SPEAKER = 1 # number of speakers
N_RECORDS = MAX_DATA_LEN # number of recordings
# Assegnazione ternaria di DATA_LEN
DATA_LEN = N_SPEAKER * N_RECORDS if N_SPEAKER * N_RECORDS < MAX_DATA_LEN else MAX_DATA_LEN

MIN_SIG_LEN = 2  # only use snippets longer than 2 seconds
fs = 16000  # sampling rate
sig_len = 2048  # length of snippet used for tdoa estimation
batch_size = N_SPEAKER # Batch size for DataLoader
speaker_index = [61, 121, 237, 260] # [] to select all speakers 


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
snr = 10 # dB
c = 343  # speed of sound in m/s



# source lacation points random generated in meters
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
tetraural_mic_dir = [
    CardioidFamily(orientation=DirectionVector(azimuth=45, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=135, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=315, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=235, colatitude=135, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    ]
# -------- square
square_mic_locs = np.array([[1.4, 1.15, 1.8], [1.6, 1.15, 1.8], [1.4, 1.35, 1.8], [1.6, 1.35, 1.8]]).T
square_mic_dir = [
    CardioidFamily(orientation=DirectionVector(azimuth=225, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=315, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=135, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    CardioidFamily(orientation=DirectionVector(azimuth=45, colatitude=90, degrees=True), pattern_enum=DirectivityPattern.CARDIOID),
    ]
# -------- circular
circular_mic_rotation = 0
circular_mic_center = np.array(room_dim) / 2
circular_mic_colatitude = 90
circular_mic_num_mic = 8
circular_mic_pattern = DirectivityPattern.CARDIOID
circular_mic_orientation = DirectionVector(azimuth=circular_mic_rotation, colatitude=circular_mic_colatitude, degrees=True)
circular_mic_directivity = CardioidFamily(orientation=circular_mic_orientation, pattern_enum=circular_mic_pattern)
circular_mic_array = pra.beamforming.circular_microphone_array_xyplane(
    center=circular_mic_center,
    M= circular_mic_num_mic,
    phi0=circular_mic_rotation,
    radius=10e-2,
    fs=fs,
    directivity=circular_mic_directivity,
)
circular_mic_locs = circular_mic_array.R

