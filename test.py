import torch
import torchaudio
import librosa

from transforms import SpecAugment
from functional import warp_axis_torch, mask_along_axis
from utils.plots import plot_spectrogram


def create_specgram(audio_path: str, num_batch: int):
    torch_data, sr  = torchaudio.load(audio_path)
    # Convert audio array to mel spectrogram
    librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                                n_fft=2048,
                                                win_length=1024,
                                                sr=sr)

    tensor_mel = torch.tensor(librosa_mel)
    stacked_mel = tensor_mel.repeat((num_batch,1,1))
    return stacked_mel



batch_specgram = create_specgram("audio_data/lex_30.wav", 100)

spec = SpecAugment(
        warp_axis = 2,
        warp_w = 500,
        freq_mask_num = 3,
        freq_mask_param = 0,
        freq_mask_p = 0.1,
        time_mask_num = 3,
        time_mask_param = 0,
        time_mask_p = .05
)
for _ in range(500):
    augmented = spec(batch_specgram)
#plot_spectrogram(augmented[0])
