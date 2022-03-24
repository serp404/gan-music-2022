import os
import torch
from torch import Tensor
import torchaudio
import typing as tp

from music_project.common.melspec import MelSpectrogram


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.wav_files = [file for file in os.listdir(data_path) if file.endswith(".wav")]
        self.mel_converter = MelSpectrogram()

    def __getitem__(self, idx: int) -> tp.Tuple[Tensor, Tensor]:
        '''
            Returns tuple of Tensors: [waveform, melspectrogram]
            - waveform: shape [1, wav_len]
            - melspecpectrogram: shape [1, 80, mel_len]
        '''
        audio_path = os.path.join(self.data_path, self.wav_files[idx])
        waveform, _ = torchaudio.load(audio_path)
        melspec = self.mel_converter(waveform)
        return waveform, melspec

    def __len__(self) -> int:
        return len(self.wav_files)
