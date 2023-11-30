# librispeech-ssl-datasets
**Sound Source Localization**
This project aims to create simulated source sound localization dataset in customizable indoor room with random speaker position and speaker from LibriSpeech.
This project is also about sound source localization using various Direction of Arrival (DOA) algorithms. 
<center><img src="./assets/librispeech.png" width="100%" /></center>
<center> + </center>                      
<center><img src="./assets/pyroomacoustics.png" width="100%"/></center>   

Thanks to [PyRoomsAcoustics](https://github.com/LCAV/pyroomacoustics)
, [PyTorch](https://github.com/pytorch/pytorch) to manage [LibriSpeech](https://www.openslr.org/12) of [OpenSLR](https://www.openslr.org/index.html) and [NGCC-PHAT repo](https://github.com/axeber01/ngcc) for inspiration to solve audio acquisition dataset in this way.


## Libraries Used
- **numpy:** Used for numerical computations.
- **pyroomacoustics:** Used for simulating room acoustics.
- **torch:** Used for deep learning operations.

## Uses Example:
See example_how_to_use.ipynb

## How to Run

### To run the main script, use the following command:
python main.py



# Code Overview
## Configuration of Dataset based on Environment Settings [cfg.py](./cfg.py)
- The selected code is a configuration file (cfg.py) for a room simulation. It sets up various parameters for the simulation.

    - **dx, dy, dz** define the dimensions of the room in meters.
    - **room_dim** is a list that contains the dimensions of the room.
    - **xyz_min** and **xyz_max** define the minimum and maximum coordinates for the room.
    - **t60** is the reverberation time of the room in seconds.
    - **snr** is the signal-to-noise ratio in decibels.
    - **c** is the speed of sound in meters per second.
    - **source_locs** is an array of random source locations within the room.
    - **mic_config** is a string that defines the microphone configuration. It can be 'binaural', 'triaural', 'tetraural', 'square', or 'circular'.
    -binaural_mic_locs, triaural_mic_locs, and tetraural_mic_locs are arrays that define the locations of the microphones in each configuration.
    - binaural_mic_dir, triaural_mic_dir are lists that define the direction of the microphones in binaural and triaural configurations respectively. They use the CardioidFamily class to define the directivity pattern and orientation of the microphones.

- The selected code is part of a configuration file (cfg.py) and it sets up parameters for Direction of Arrival (DOA) algorithms.

    - **algo_names** is a list of the DOA algorithms to be used. In this case, the algorithms are 'SRP', 'MUSIC', 'TOPS', 'NormMUSIC', and 'WAVES'. These are different methods used to estimate the direction of a source of sound.
    - **nfft** is the size of the Fast Fourier Transform (FFT) to be used. FFT is a method for computing the Discrete Fourier Transform (DFT) of a sequence, and it's used in signal processing. The size of the FFT is a parameter that can affect the resolution and computation time of the FFT.
    - **freq_range** is a list that defines the frequency range for the DOA estimation. In this case, the range is from 300 to 16000 Hz. This means that the algorithms will try to estimate the direction of arrival of sounds within this frequency range.

## Dataset Creation and Simulation [data.py](./data.py)
The file data.py contains two main classes: LibriSpeechLocationsDataset and RoomSimulator.

## LibriSpeechLocationsDataset class
It extends the LIBRISPEECH class and has two main methods:

1. __init__(self, source_locs, split): This is the class constructor. It takes source positions and dataset split type as arguments. It initializes the object and sets self.source_locs to the provided source positions.

2. __getitem__(self, n: int): This method allows accessing the object's elements as if it were an array. It takes an index n and returns a tuple with the waveform, sample rate, transcript, speaker ID, utterance number, source position, and a seed.

## RoomSimulator Class
This class is designed to simulate the signal propagation from a source to microphone positions, given a batch of LibrispeechLocation samples.

1. The __init__ method is the class constructor. This method is called when an instance of the class is created. It initializes the object with a set of configuration parameters defined in cfg. These include signal length, frequency bounds, sampling frequency, room dimensions, reverberation time, microphone configuration, signal-to-noise ratio, and microphone positions and directions.

2. The remove_silence method removes silent parts from the input signal. This is done by segmenting the signal into clips, removing clips below a certain decibel threshold (defined by top_db), and then concatenating the remaining clips.

3. The create_simulation method creates a binaural simulation using the pyroomacoustics library. This method creates a room, adds an array of microphones to the room, adds a source to the room for each signal in signals, and then simulates the room.

4. The pad_sequence method makes all tensors in a batch the same length by zero-padding.

5. The __call__ method is called when the object is "called" as a function. This method takes a batch of data, removes silence from each waveform, creates a simulation for each waveform, and then aggregates the simulated tensors into a batch. It returns the batch tensors, source positions, and microphone positions.