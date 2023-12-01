from matplotlib import pyplot as plt
from torchaudio.datasets import LIBRISPEECH
import pyroomacoustics as pra
import numpy as np
from typing import Tuple
from torch import Tensor
import torch
from librosa.effects import split

import cfg

def remove_silence(signal, top_db=20, frame_length=2048, hop_length=512):
        '''
        Remove silence from speech signal
        '''
        signal = signal.squeeze()
        clips = split(signal, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
        output = []
        for ii in clips:
            start, end = ii
            output.append(signal[start:end])

        return torch.cat(output)


def cartesian_to_polar(x, y, z):
    """
    Converte le coordinate cartesiane in coordinate polari.

    :param x: Coordinata x.
    :param y: Coordinata y.
    :param z: Coordinata z.
    :return: Coordinate polari (r, theta, phi).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi



class LibriSpeechLocationsDataset(LIBRISPEECH):
    '''
    Class of LibriSpeech recordings. Each recording is annotated with a speaker location.
    '''

    def __init__(self, source_locs, split):
        super().__init__("./", url=split, download=True)

        self.source_locs = source_locs

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int, int, float, int]:

        source_loc = self.source_locs[n]
        seed = n
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_number = super().__getitem__(n)
        return (waveform, sample_rate, transcript, speaker_id, utterance_number), source_loc, seed



class RoomSimulator(object):
    '''
    Given a batch of LibrispeechLocation samples, simulate signal
    propagation from source to the microphone locations.
    '''

    def __init__(self):
        self.N = cfg.sig_len
        self.lower_bound = cfg.lower_bound
        self.upper_bound = cfg.upper_bound
        self.fs = cfg.fs
        self.room_dim = cfg.room_dim
        self.t60 = cfg.t60
        self.mic_config = cfg.mic_config
        self.snr = cfg.snr
        self.binaural_mic_locs = cfg.binaural_mic_locs
        self.binaural_mic_dir = cfg.binaural_mic_dir
        self.triaural_mic_locs = cfg.triaural_mic_locs
        self.triaural_mic_dir = cfg.triaural_mic_dir
        self.tetraural_mic_locs = cfg.tetraural_mic_locs
        self.tetraural_mic_dir = cfg.tetraural_mic_dir
        self.square_mic_locs = cfg.square_mic_locs
        self.square_mic_dir = cfg.square_mic_dir
        self.circular_mic_locs = cfg.circular_mic_locs
        self.circular_mic_array = cfg.circular_mic_array
        


    def remove_silence(self, signal, top_db=20, frame_length=2048, hop_length=512):
        '''
        Remove silence from speech signal
        '''
        signal = signal.squeeze()
        clips = split(signal, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
        output = []
        for ii in clips:
            start, end = ii
            output.append(signal[start:end])

        return torch.cat(output)
    

    def create_simulation(self, source_locs, signals):
        '''
        Create a binaural simulation using pyroomacoustics.
        '''

        # 0 Create room
        e_absorption, max_order = pra.inverse_sabine(self.t60, self.room_dim)
        room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=pra.Material(e_absorption), max_order=max_order)
        # 1 Add microphone to room
        if self.mic_config == 'binaural':
            room.add_microphone_array(self.binaural_mic_locs, directivity=self.binaural_mic_dir)
            self.mic_locs = self.binaural_mic_locs
        elif self.mic_config == 'triaural':
            room.add_microphone_array(self.triaural_mic_locs, directivity=self.triaural_mic_dir)
            self.mic_locs = self.triaural_mic_locs
        elif self.mic_config == 'tetraural':
            room.add_microphone_array(self.tetraural_mic_locs, directivity=self.tetraural_mic_dir)
            self.mic_locs = self.tetraural_mic_locs
        elif self.mic_config == 'square':
            room.add_microphone_array(self.square_mic_locs, directivity=self.square_mic_dir)
            self.mic_locs = self.square_mic_locs
        elif self.mic_config == 'circular':
            room.add_microphone_array(self.circular_mic_array)
            self.mic_locs = self.circular_mic_locs
        cfg.c = room.c
        source_locations=[]
        # 2 Add source to room
        for i, signal in enumerate(signals):
            source_loc = source_locs[i]
            source_locations.append(source_loc)
            room.add_source(source_loc, signal=signal)

        # 3 Simulate room
        room.simulate(snr=self.snr)

        return room.mic_array.signals.T, source_locations, self.mic_locs


    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch


    def __call__(self, batch):
            # A data tuple has the form:
            # waveform, sample_rate, label, speaker_id, utterance_number
            tensors1 = []
            targets = []
            # Gather in lists, and encode labels as indices
            with torch.no_grad():
                for (waveform, sample_rate, label, speaker_id, utterace_number), source_loc, seed in batch:

                    waveform = waveform.squeeze()
                    signal = self.remove_silence(waveform, frame_length=self.N)
                    simulated_signals, source_locs, mic_locs = self.create_simulation(source_locs=[source_loc],signals=[signal])
                    # if self.train:
                    #     start_idx = torch.randint(self.lower_bound, self.upper_bound - self.N - 1, (1,))
                    # else:
                    #     start_idx = self.lower_bound
                    #     end_idx = start_idx + self.N
                    #     simulated_signals = simulated_signals[start_idx:end_idx]                    
                    # convert simulated_signals to np array
                    simulated_signals = np.array(simulated_signals)
                    # convert source_locs to np array
                    source_locs = np.array(source_locs)
                    # Group the list of tensors into a batched tensor
                    tensors1 += [torch.as_tensor(simulated_signals, dtype=torch.float)]
                    targets += [torch.as_tensor(source_locs, dtype=torch.float)]

            # Group the list of tensors into a batched tensor
            tensors1 = self.pad_sequence(tensors1).unsqueeze(1)

            return tensors1, targets, mic_locs
        

