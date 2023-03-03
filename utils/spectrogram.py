import matplotlib.pyplot as plt
import librosa
import torch
import torchaudio


def create_specgram(audio_path: str, batch_sz: int):
    torch_data, sr  = torchaudio.load(audio_path)
    # Convert audio array to mel spectrogram
    librosa_mel = librosa.feature.melspectrogram(y=torch_data[0].numpy(),
                                                n_fft=2048,
                                                win_length=1024,
                                                sr=sr)

    tensor_mel = torch.tensor(librosa_mel)
    stacked_mel = tensor_mel.repeat((batch_sz,1,1))
    return stacked_mel


def plot_specgram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1, figsize=(8,6))
  fig.figsize = (15, 15)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=True)