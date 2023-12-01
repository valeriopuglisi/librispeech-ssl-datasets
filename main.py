# %%
from torchaudio.datasets import LIBRISPEECH
import torchaudio
import pyroomacoustics as pra
import numpy as np
import torch
from tqdm import tqdm
import cfg
from data import remove_silence, LibriSpeechLocationsDataset, RoomSimulator, cartesian_to_polar
from pyroomacoustics.doa import circ_dist


# %%
print(f"room dimensions: {cfg.room_dim}")
print(f"microphone configuration: {cfg.mic_config}")
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
# create room simulator
room_simulator = RoomSimulator()

# %%
# create dataloader
dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
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
        X = X.transpose([2, 1, 0])

        
        spatial_resp = dict()

        # conversion from cartesian to polar coordinates
        r, theta, phi = cartesian_to_polar(source_locs[0] - (cfg.dx/2), source_locs[1] - (cfg.dy/2), source_locs[2]-(cfg.dz/2))
        azimuth = [theta]


        # loop through algos and localize
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
            recovered_azimuth = doa.azimuth_recon / np.pi * 180.0
            real_azimuth = theta / np.pi * 180.0
            error_azimuth = circ_dist(theta, doa.azimuth_recon) / np.pi * 180.0
            # print(algo_name)
            # print("  Recovered azimuth:", recovered_azimuth ,"degrees")
            # print("  Real azimuth:", real_azimuth, "degrees")
            # print("  Error:", error_azimuth, "degrees")
            # print(f" type(error_azimuth): {type(error_azimuth.numpy())}")

            #algo_names = ['SRP', 'MUSIC', 'TOPS','NormMUSIC', 'WAVES']
            if algo_name == "SRP":
                srp_azimuth_errors.append(error_azimuth.numpy())  
            elif algo_name == "MUSIC":
                music_azimuth_errors.append(error_azimuth.numpy())
            elif algo_name == "TOPS":
                tops_azimuth_errors.append(error_azimuth.numpy())  
            elif algo_name == "NormMUSIC":
                norm_music_azimuth_errors.append(error_azimuth.numpy())
            elif algo_name == "WAVES":  
                waves_azimuth_errors.append(error_azimuth.numpy())  
            
        pbar.update(pbar_update)

print(f"----type(srp_azimuth_errors): {type(srp_azimuth_errors)}")


# convert to numpy arrays 
srp_azimuth_errors = np.array(srp_azimuth_errors)
music_azimuth_errors = np.array(music_azimuth_errors)
tops_azimuth_errors = np.array(tops_azimuth_errors)
norm_music_azimuth_errors = np.array(norm_music_azimuth_errors)
waves_azimuth_errors = np.array(waves_azimuth_errors)

# calutate mean error for each algorithm
srp_mean_error = np.mean(srp_azimuth_errors)
music_mean_error = np.mean(music_azimuth_errors)
tops_mean_error = np.mean(tops_azimuth_errors)
norm_music_mean_error = np.mean(norm_music_azimuth_errors)
waves_mean_error = np.mean(waves_azimuth_errors)

# write mean arrays into a pandas dataframe
import pandas as pd
df = pd.DataFrame({'SRP_mean_error': srp_mean_error, 
                   'MUSIC_mean_error': music_mean_error, 
                   'TOPS_mean_error': tops_mean_error, 
                   'NormMUSIC_mean_error': norm_music_mean_error, 
                   'WAVES_mean_error': waves_mean_error}, index=[0])

# save datafreame into a csv file inserting the microphone configuration in the name
df.to_csv('./'+str(cfg.mic_config)+'_localization_mean_error.csv')

# create a plot from this dataframe setting vertical orientation of x label, range from 0 to 360 for y and save it
import matplotlib.pyplot as plt
ax = df.plot.bar(rot=0, ylim=(0, 360))
ax.set_ylabel('Mean error (degrees)')
ax.set_xlabel('Algorithms')
ax.set_title('Mean error for each localization algorithm in '+str(cfg.mic_config)+' configuration')
plt.savefig('./'+str(cfg.mic_config)+'_localization_mean_error.png')



