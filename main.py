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
import matplotlib.pyplot as plt

# %%
print(f"room dimensions: {cfg.room_dim}")
print(f"microphone configuration: {cfg.mic_config}")
# create datasets
print(f"len(source_locs_train): {len(cfg.source_locs)}")
dataset = LibriSpeechLocationsDataset(cfg.source_locs, split="test-clean")
print(f"len(dataset): {len(dataset)}")

# %%
# remove silence and keep only waveforms longer than MIN_SIG_LEN seconds
print("Removing silence and keeping only waveforms longer than MIN_SIG_LEN seconds")
valid_idx = [i if len(remove_silence(waveform, frame_length=cfg.sig_len)) > cfg.fs * cfg.MIN_SIG_LEN else None for i, ((waveform, sample_rate,transcript, speaker_id, utterance_number), pos, seed) in enumerate(dataset)]
inds = [i for i in valid_idx if i is not None]
print("Silence removed")
# %%

# %%
dataset = torch.utils.data.dataset.Subset(dataset, inds)
print('Total data set size after removing silence: ' + str(len(dataset)))

# %%
# create room simulator
room_simulator = RoomSimulator()

# %%
# create dataloader
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
    
        microphone_signals = microphone_signals.squeeze(0).squeeze(0).numpy().T     
        # print(f"microphone_signals.shape: {microphone_signals.shape}")
        # print(f"source_locs.shape: {source_locs.shape[0]}")   
        # print(f"mic_locs.shape : {mic_locs.shape}")
        # print(f"mic_locs : {mic_locs}")
        # Convert to frequency domain
        X = pra.transform.stft.analysis(microphone_signals, cfg.nfft, cfg.nfft // 2)
        X = X.transpose([2, 1, 0])
        spatial_resp = dict()

        # create plot with microphone and source locations put limit as cfg.room_dim
        # fig, ax = plt.subplots()
        # ax.scatter(mic_locs[0], mic_locs[1], marker='o', color='r', label='Microphone')
        # ax.scatter(source_locs[:,0], source_locs[:,1], marker='x', color='b', label='Source')
        # ax.set_xlim(0, cfg.room_dim[0])
        # ax.set_ylim(0, cfg.room_dim[1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.legend()
        # plt.savefig('./'+str(cfg.mic_config)+'_microphone_source_locations.png')
        # plt.close(fig)
      
        # loop through algos and localize
        for algo_name in cfg.algo_names:
            try:
                # Construct the new DOA object
                # the max_four parameter is necessary for FRIDA only
                doa = pra.doa.algorithms[algo_name](mic_locs, cfg.fs, cfg.nfft, c=cfg.c, num_src=source_locs.shape[0], max_four=4)

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
                

                azimuth = []
                for i, source_loc in enumerate(source_locs):    
                    # conversion from cartesian to polar coordinates
                    x = source_loc[0] - (cfg.dx/2)
                    y = source_loc[1] - (cfg.dy/2)
                    z = source_loc[2] - (cfg.dz/2)
                    r, theta, phi = cartesian_to_polar(x , y, z)
                    if theta < 0:
                        theta = theta + 2*np.pi
                    azimuth.append(theta)
            
                # sort azimuths 
                azimuth = np.sort(np.array(azimuth))
                doa_azimuth_recon = np.sort(doa.azimuth_recon)
                recovered_azimuth = doa_azimuth_recon / np.pi * 180.0
                real_azimuth = azimuth / np.pi * 180.0
                error_azimuth = circ_dist(azimuth, doa_azimuth_recon) / np.pi * 180.0
                # print(algo_name)
                # print("  Real azimuth:", real_azimuth, "degrees")
                # print("  Recovered azimuth:", recovered_azimuth ,"degrees")
                # print("  Error:", error_azimuth, "degrees")

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
            except Exception as e:
                print(f"Exception for {algo_name} with message {e}")
                continue    
        pbar.update(pbar_update)


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



