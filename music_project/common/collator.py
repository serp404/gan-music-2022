from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import typing as tp

from music_project.common.melspec import MelSpectrogram, MelSpectrogramConfig
from music_project.common.utils import wav_splitter
from music_project.config import TaskConfig


featurizer = MelSpectrogram()
max_wav_len = TaskConfig.max_wav_len
mel_pad_val = MelSpectrogramConfig.pad_value


def split_collate_fn(
    samples: tp.List[tp.Tuple[Tensor, Tensor]]
) -> tp.Tuple[Tensor, Tensor]:

    wavs, mels = [], []
    for wav_org, _ in samples:
        for wav_part, mels_part in wav_splitter(
            wav_org, featurizer, max_wav_len
        ):
            wavs.append(wav_part.squeeze(dim=0))
            mels.append(mels_part.squeeze(dim=0).transpose(-1, -2))

    wavs_tensor = pad_sequence(wavs, batch_first=True)
    mels_tensor = pad_sequence(
        mels, batch_first=True, padding_value=mel_pad_val
    ).transpose(-1, -2)

    return wavs_tensor, mels_tensor


def base_collate_fn(
    samples: tp.List[tp.Tuple[Tensor, Tensor]]
) -> tp.Tuple[Tensor, Tensor]:

    wavs = [el.squeeze(dim=0) for el, _ in samples]
    mels = [el.squeeze(dim=0).transpose(-1, -2) for _, el in samples]

    wavs_tensor = pad_sequence(wavs, batch_first=True)
    mels_tensor = pad_sequence(
        mels, batch_first=True, padding_value=mel_pad_val
    ).transpose(-1, -2)

    return wavs_tensor, mels_tensor
