# %%
from torchaudio.datasets import LIBRISPEECH
import torchaudio
from IPython.display import Audio, display
import pyroomacoustics as pra
import numpy as np
from typing import Tuple
from torch import Tensor
import torch
import random
from tqdm import tqdm
from librosa.effects import split
import cfg
from data import remove_silence, LibriSpeechLocationsDataset, RoomSimulator, cartesian_to_polar
from pyroomacoustics.doa import spher2cart, circ_dist


# %%
# create datasets
print(f"len(source_locs_train): {len(cfg.source_locs)}")
dataset = LibriSpeechLocationsDataset(cfg.source_locs, split="test-clean")
print(f"len(dataset): {len(dataset)}")


# %%
# Get a sample of created dataset
# print('Total data set size: ' + str(len(dataset))) 
# (waveform, sample_rate, transcript, speaker_id, utterance_number), pos, seed = dataset[0]
# transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=cfg.fs)
# transformed = transform(waveform)
# # write code to play transformed with Audio
# Audio(transformed.numpy(), rate=cfg.fs)

# %%
# remove silence and keep only waveforms longer than MIN_SIG_LEN seconds
print("Removing silence and keeping only waveforms longer than MIN_SIG_LEN seconds")
valid_idx = [i if len(remove_silence(waveform, frame_length=cfg.sig_len)) > cfg.fs * cfg.MIN_SIG_LEN else None for i, ((waveform, sample_rate,transcript, speaker_id, utterance_number), pos, seed) in enumerate(dataset)]
inds = [i for i in valid_idx if i is not None]
print("Silence removed")

# %%
# print(f"len(valid_idx): {len(valid_idx)} -  valid_idx: {valid_idx}")
# print(f"len(inds): {len(inds)} - valid inds: {inds}")

# %%

dataset = torch.utils.data.dataset.Subset(dataset, inds)
print('Total data set size after removing silence: ' + str(len(dataset)))


# %%
# (waveform, sample_rate, transcript, speaker_id, utterance_number), pos, seed = dataset[0]
# transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=cfg.fs)
# transformed = transform(waveform)
# print(f"waveform.shape: {waveform.shape}")
# print(f"sample_rate: {sample_rate}")
# print(f"transcript: {transcript}")
# print(f"speaker_id: {speaker_id}")
# print(f"utterance_number: {utterance_number}")
# print(f"pos: {pos}")
# print(f"seed: {seed}")
# # write code to play transformed with Audio
# Audio(transformed.numpy(), rate=cfg.fs)

# %%
# print(f"transformed.shape: {transformed.shape}")
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)


# %%
# pbar_update = cfg.batch_size
# with tqdm(total=len(dataset)) as pbar:
#     for batch_idx, sample in enumerate(train_loader):
#         (waveform, sample_rate, transcript, speaker_id, utterance_number), pos, seed = sample
#         print(f"batch_idx: {batch_idx}")
#         print(f"waveform.shape: {waveform.shape}")
#         print(f"sample_rate: {sample_rate}")
#         print(f"transcript: {transcript}")
#         print(f"speaker_id: {speaker_id}")
#         print(f"utterance_number: {utterance_number}")
#         print(f"pos: {pos}")
#         print(f"seed: {seed}")
#         pbar.update(pbar_update)

# %%
# Explain shortly What is RoomSimulator: 
room_simulator = RoomSimulator()



# %%
dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=room_simulator,
)


# %%
# with tqdm(total=len(dataset_loader)) as pbar:
#algo_names = ['SRP', 'MUSIC', 'TOPS','NormMUSIC', 'WAVES']
srp_azimuth_errors = []
music_azimuth_errors = []
tops_azimuth_errors = []
norm_music_azimuth_errors = []
waves_azimuth_errors = []

pbar_update = cfg.batch_size

with tqdm(total=len(dataset_loader)) as pbar:
    for batch_idx, (microphone_signals, source_locs, mic_locs) in enumerate(dataset_loader):
            
        source_locs = source_locs[0].squeeze(0)
        microphone_signals = microphone_signals.squeeze(0).squeeze(0).numpy().T     

        # Convert to frequency domain
        X = pra.transform.stft.analysis(microphone_signals, cfg.nfft, cfg.nfft // 2)
        print(f"X.shape: {X.shape}")
        X = X.transpose([2, 1, 0])

        
        spatial_resp = dict()

        # Conversione alle coordinate polari
        r, theta, phi = cartesian_to_polar(source_locs[0] - (cfg.dx/2), source_locs[1] - (cfg.dy/2), source_locs[2]-(cfg.dz/2))
        print(f"theta:{theta}")

        azimuth = [theta]


        # loop through algos
        for algo_name in cfg.algo_names:
            # Construct the new DOA object
            # the max_four parameter is necessary for FRIDA only
            doa = pra.doa.algorithms[algo_name](mic_locs, cfg.fs, cfg.nfft, c=cfg.c, num_src=1, max_four=4)

            # this call here perform localization on the frames in X
            doa.locate_sources(X, freq_range=cfg.freq_range)
            
            # store spatial response
            if algo_name == 'FRIDA':
                    spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
            else:
                    spatial_resp[algo_name] = doa.grid.values
                    
            # normalize   
            min_val = spatial_resp[algo_name].min()
            max_val = spatial_resp[algo_name].max()
            spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)
            print(algo_name)
            recovered_azimuth = doa.azimuth_recon / np.pi * 180.0
            real_azimuth = theta / np.pi * 180.0
            error_azimuth = circ_dist(theta, doa.azimuth_recon) / np.pi * 180.0
            print("  Recovered azimuth:",  "degrees")
            print("  Real azimuth:", real_azimuth, "degrees")
            print("  Error:", error_azimuth, "degrees")

            #algo_names = ['SRP', 'MUSIC', 'TOPS','NormMUSIC', 'WAVES']
            if algo_name == "SRP":
                srp_azimuth_errors.append(error_azimuth)  
            elif algo_name == "MUSIC":
                music_azimuth_errors.append(error_azimuth)
            elif algo_name == "TOPS":
                tops_azimuth_errors.append(error_azimuth)  
            elif algo_name == "NormMUSIC":
                norm_music_azimuth_errors.append(error_azimuth)
            elif algo_name == "WAVES":  
                waves_azimuth_errors.append(error_azimuth)  
        pbar.update(pbar_update)

