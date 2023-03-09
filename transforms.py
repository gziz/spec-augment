import torch
from torch import Tensor

import functional as F

class SpecAugmentTransform(torch.nn.Module):
    """
    :cite:t:`Park_2019`.

    :param warp_axis:
        Axis where the warp takes place (1: freq, 2: time)
    :param max_warp_length:
        Boundaries where warp takes place (max_warp_length, N - max_warp_length), (W in paper)
    :param num_freq_mask:
        Number of masks to apply to the frequency axis, (mF in paper)
    :param freq_max_mask_length:
        Max length of any individual frequency mask, (F in paper)
    :param freq_mask_max_proportion:
        Max proportion that any individual freq mask can have
    :param num_time_mask:
        Number of masks to apply to the time axis, (mT in paper)
    :param time_max_mask_length:
        Max length of any individual time mask, (T in paper)
    :param time_mask_max_proportion:
        Max proportion that any individual time mask can have, (p in paper)
    """

    def __init__(
        self,
        warp_axis: int = 2,
        max_warp_length: int = 0,
        num_freq_mask: int = 0,
        freq_max_mask_length: int = 0,
        freq_mask_max_proportion: float = 1.0,
        num_time_mask: int = 0,
        time_max_mask_length: int = 0,
        time_mask_max_proportion: float = 1.0,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.warp_axis = warp_axis
        self.max_warp_length = max_warp_length
        self.num_freq_mask = num_freq_mask
        self.freq_max_mask_length = freq_max_mask_length
        self.freq_mask_max_proportion = freq_mask_max_proportion
        self.num_time_mask = num_time_mask
        self.time_max_mask_length = time_max_mask_length
        self.time_mask_max_proportion = time_mask_max_proportion
        self.mask_value = mask_value

    def forward(self, specgram: Tensor) -> Tensor:
        """
        :param specgram:
            Tensor to augment. *Shape:* :math:`(N,F,T)`, or :math:`(F,T)` when
            unbatched, where :math:`N` is the batch size, :math:`F` is the
            frequency axis, and :math:`T` is the time axis.

        :returns:
            Augmented spectrogram. *Shape:* Same as input.
        """

        if not self.training:
            return specgram

        return F.spec_augment(
            specgram,
            self.warp_axis,
            self.max_warp_length,
            self.num_freq_mask,
            self.freq_max_mask_length,
            self.freq_mask_max_proportion,
            self.num_time_mask,
            self.time_max_mask_length,
            self.time_mask_max_proportion,
            self.mask_value,
        )
    
    @staticmethod
    def libri_speech_basic(self) -> None:
        """Sets parameters for the LibriSpeech basic level."""
        self.max_warp_length = 80
        self.freq_max_mask_length, self.num_freq_mask = 27, 1
        self.time_max_mask_length, self.num_time_mask = 100, 1
        self.time_mask_max_proportion = 1.0

    @staticmethod
    def libri_speech_double(self) -> None:
        """Sets parameters for the LibriSpeech double level."""
        self.max_warp_length = 80
        self.freq_max_mask_length, self.num_freq_mask = 27, 2
        self.time_max_mask_length, self.num_time_mask = 100, 2
        self.time_mask_max_proportion = 1.0

    @staticmethod
    def switchboard_mild(self) -> None:
        """Sets parameters for the SwitchBoard mild level."""
        self.max_warp_length = 40
        self.freq_max_mask_length, self.num_freq_mask = 15, 2
        self.time_max_mask_length, self.num_time_mask = 70, 2
        self.time_mask_max_proportion = 0.2

    @staticmethod
    def switchboard_strong(self) -> None:
        """Sets parameters for the SwitchBoard strong level."""
        self.max_warp_length = 40
        self.freq_max_mask_length, self.num_freq_mask = 27, 2
        self.time_max_mask_length, self.num_time_mask = 70, 2
        self.time_mask_max_proportion = 0.2
