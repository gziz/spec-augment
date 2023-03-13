import random
import torch
from torch import Tensor


def spec_augment(
    specgram: Tensor,
    stretch_axis: int = 2,
    max_stretch_length: int = 0,
    num_freq_mask: int = 0,
    freq_max_mask_length: int = 0,
    freq_mask_max_proportion: float = 0.0,
    num_time_mask: int = 0,
    time_max_mask_length: int = 0,
    time_mask_max_proportion: float = 0.0,
    mask_value: float = 0.0,
) -> Tensor:
    """Apply data augmentation SpecAugment to a spectrogram tensor
    as described in :cite:t:`Park_2019`.

    :param specgram:
        Tensor to augment. *Shape:* :math:`(N,F,T)`, or :math:`(F,T)` when
        unbatched, where :math:`N` is the batch size, :math:`F` is the
        frequency axis, and :math:`T` is the time axis.
    :param stretch_axis:
        Axis where the stretch takes place (1: freq, 2: time)
    :param max_stretch_length:
        Boundaries where stretch takes place (max_stretch_length, N - max_stretch_length), (W in paper)
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

    :returns:
        Augmented spectrogram. *Shape:* Same as input.

    .. note::
    The paper implements a time warp while this specAugment implements a time stretch
    (or stretch to the specified param stretch_axis).
    """

    specgram = specgram.clone()

    if specgram.dim() == 2:
        specgram = specgram.unsqueeze(0)

    specgram = _stretch_along_axis(specgram, stretch_axis, max_stretch_length)
    specgram = _mask_along_axis(
        specgram, 1, num_freq_mask, freq_max_mask_length, freq_mask_max_proportion, mask_value
    )
    specgram = _mask_along_axis(
        specgram, 2, num_time_mask, time_max_mask_length, time_mask_max_proportion, mask_value
    )
    return specgram


def _stretch_along_axis(specgram: Tensor, axis: int, max_stretch_length: int) -> Tensor:
    """Apply a stretch to a spectrogram along a specified axis.

    :param specgram:
        Tensor to stretch. *Shape:* :math:`(N,F,T)`, where :math:`N`
        is the batch size, :math:`F` is the frequency axis, and
        :math:`T` is the time axis.
    :param axis:
        Axis where the stretch takes place
    :param max_stretch_length:
        Boundary of time steps where the stretch takes place (max_stretch_length, stretched_dim_size - max_stretch_length)

    :returns:
        Stretched tensor spectrogram of dimensions (batch, freq, time)

    .. note::
        stretch takes place between the max_stretch_length boundaries, starting from point w0, the stretch
        direction can be negative or positive, depending on the randomly chosen distance w.
    """

    if max_stretch_length == 0:
        return specgram

    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    stretched_dim_size = specgram.shape[axis]
    non_stretched_dim_size = specgram.shape[1 if axis == 2 else 2]

    if 2 * max_stretch_length >= stretched_dim_size:
        raise ValueError(
            f"max_stretch_length param {max_stretch_length} must be smaller than half the size of the stretched axis {stretched_dim_size}")

    w0 = random.randint(max_stretch_length, stretched_dim_size - max_stretch_length)
    w = random.randint(-max_stretch_length, max_stretch_length + 1)

    if axis == 1:
        lower, upper = specgram[:, :w0, :], specgram[:, w0:, :]
        lower_sz = (w0 + w, non_stretched_dim_size)
        upper_sz = (stretched_dim_size - w0 - w, non_stretched_dim_size)
    else:
        lower, upper = specgram[:, :, :w0], specgram[:, :, w0:]
        lower_sz = (non_stretched_dim_size, w0 + w)
        upper_sz = (non_stretched_dim_size, stretched_dim_size - w0 - w)

    # interpolate receives 4D: (batch, channel, freq, time)
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)

    lower = torch.nn.functional.interpolate(lower, size=lower_sz, mode="bilinear")
    upper = torch.nn.functional.interpolate(upper, size=upper_sz, mode="bilinear")

    lower = lower.squeeze(1)
    upper = upper.squeeze(1)

    return torch.cat([lower, upper], dim=axis)


def _mask_along_axis(
    specgram: Tensor,
    axis: int,
    num_masks: int,
    max_mask_length: int,
    max_mask_proportion: float = 0.0,
    mask_value: float = 0.0,
) -> Tensor:
    """Mask blocks of channels along a spectrogram's axis.

    :param specgram:
        Tensor to mask. *Shape:* :math:`(N,F,T)`, where :math:`N`
        is the batch size, :math:`F` is the frequency axis, and
        :math:`T` is the time axis.
    :param axis:
        Masking is applied (1: freq, 2: time)
    :param num_masks:
        Number of masks
    :param max_mask_length:
        Max length allowed for each individual mask.
    :param max_mask_proportion:
        Max proportion of masked rows/cols for each individual mask.
    :param mask_value:
        Value to fill the masks

    :returns:
        Masked spectrogram. *Shape:* Same as input.

    .. note::
        The length of the mask is randomly chosen, with a cap on max_mask_length.
    """

    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    max_mask_length = min(max_mask_length, int(specgram.shape[axis] * max_mask_proportion))
    if max_mask_length < 1:
        return specgram

    mask_size = random.randint(0, max_mask_length)

    for _ in range(num_masks):
        mask_start = random.randint(0, specgram.shape[axis] - mask_size)
        mask_end = mask_start + mask_size

        if axis == 1:
            specgram[:, mask_start:mask_end, :] = mask_value
        else:
            specgram[:, :, mask_start:mask_end] = mask_value

    return specgram


class SpecAugmentTransform(torch.nn.Module):
    """Apply data augmentation SpecAugment to a spectrogram tensor
    as described in :cite:t:`Park_2019`.

    :param stretch_axis:
        Axis where the stretch takes place (1: freq, 2: time)
    :param max_stretch_length:
        Boundaries where stretch takes place (max_stretch_length, N - max_stretch_length), (W in paper)
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
        stretch_axis: int = 2,
        max_stretch_length: int = 0,
        num_freq_mask: int = 0,
        freq_max_mask_length: int = 0,
        freq_mask_max_proportion: float = 1.0,
        num_time_mask: int = 0,
        time_max_mask_length: int = 0,
        time_mask_max_proportion: float = 1.0,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.stretch_axis = stretch_axis
        self.max_stretch_length = max_stretch_length
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

        .. note::
        The paper implements a time warp while this specAugment implements a time stretch
        (or stretch to the specified param stretch_axis).
        """

        if not self.training:
            return specgram

        return spec_augment(
            specgram,
            self.stretch_axis,
            self.max_stretch_length,
            self.num_freq_mask,
            self.freq_max_mask_length,
            self.freq_mask_max_proportion,
            self.num_time_mask,
            self.time_max_mask_length,
            self.time_mask_max_proportion,
            self.mask_value,
        )
    