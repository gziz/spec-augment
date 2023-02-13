import torch
import torchaudio
import librosa

from transforms import SpecAugment
from fairseq_spec import SpecAugmentTransform
from plots import plot_spectrogram



torch_data, sr  = torchaudio.load('lex_6.wav')
# Convert audio array to mel spectrogram
librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                             n_fft=2048,
                                             win_length=1024,
                                             sr=sr)

### Current Implementation ###
librosa_mel = torch.tensor(librosa_mel)

sAug = SpecAugment(
            time_warp_w = 150,
            freq_mask_n = 2,
            freq_mask_f = 10,
            time_mask_n = 3,
            time_mask_t = 40,
            time_mask_p = 1.0,
)
augmented = sAug(librosa_mel)

# Plot only receives 2D
if augmented.dim() == 3:
    augmented = augmented[0].squeeze()

plot_spectrogram(augmented)
