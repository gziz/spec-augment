import random
import torch
from torch import Tensor

def warp_along_axis(specgram: Tensor, axis: int, max_warp_length: int) -> Tensor:
    """
    Apply a warp axis to spectrogram.

    :param specgram:
        Tensor to warp. *Shape:* :math:`(N,F,T)`, where :math:`N`
        is the batch size, :math:`F` is the frequency axis, and
        :math:`T` is the time axis.
    :param axis:
        Axis where the warp takes place
    :param max_warp_length:
        Boundary of time steps where the warp takes place (max_warp_length, warped_dim_size - max_warp_length)

    :returns:
        max_warp_lengtharped tensor spectrogram of dimensions (batch, freq, time)

    .. note::
        Warp takes place between the max_warp_length boundaries, starting from point w0, the warp
        direction can be negative or positive, depending on the randomly chosen distance w.
    """

    if max_warp_length == 0:
        return specgram
    
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    warped_dim_size = specgram.shape[axis]
    non_warped_dim_size = specgram.shape[1 if axis == 2 else 2]

    if 2 * max_warp_length >= warped_dim_size:
        raise ValueError(
            f"max_warp_length param {max_warp_length} must be smaller than half the size of the warped axis {warped_dim_size}")

    w0 = random.randint(max_warp_length, warped_dim_size - max_warp_length)
    w = random.randint(-max_warp_length, max_warp_length + 1)
    
    if axis == 1:
        lower, upper = specgram[:, :w0, :], specgram[:, w0:, :]
        lower_sz = (w0 + w, non_warped_dim_size)
        upper_sz = (warped_dim_size - w0 - w, non_warped_dim_size)
    else:
        lower, upper = specgram[:, :, :w0], specgram[:, :, w0:]
        lower_sz = (non_warped_dim_size, w0 + w)
        upper_sz = (non_warped_dim_size, warped_dim_size - w0 - w)

    # interpolate receives 4D: (batch, channel, freq, time)
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)

    lower = torch.nn.functional.interpolate(lower, size=lower_sz, mode="bilinear")
    upper = torch.nn.functional.interpolate(upper, size=upper_sz, mode="bilinear")

    lower = lower.squeeze(1)
    upper = upper.squeeze(1)

    return torch.cat([lower, upper], dim=axis)


def mask_along_axis(
    specgram: Tensor,
    axis: int,
    num_masks: int,
    max_mask_length: int,
    max_mask_proportion: float = 0.0,
    mask_value: float = 0.0,
) -> Tensor:
    """
    Apply mask along a spectrogram's axis.

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

    mask_size = random.randint(max_mask_length)

    for _ in range(num_masks):
        mask_start = random.randint(specgram.shape[axis] - mask_size)
        mask_end = mask_start + mask_size

        if axis == 1:
            specgram[:, mask_start:mask_end, :] = mask_value
        else:
            specgram[:, :, mask_start:mask_end] = mask_value

    return specgram


def spec_augment(
    specgram: Tensor,
    warp_axis: int = 2,
    max_warp_length: int = 0,
    num_freq_mask: int = 0,
    freq_max_mask_length: int = 0,
    freq_mask_max_proportion: float = 0.0,
    num_time_mask: int = 0,
    time_max_mask_length: int = 0,
    time_mask_max_proportion: float = 0.0,
    mask_value: float = 0.0,
) -> Tensor:
    """
    Apply SpecAugment to spectrogram

    :param specgram:
        Tensor to augment. *Shape:* :math:`(N,F,T)`, or :math:`(F,T)` when
        unbatched, where :math:`N` is the batch size, :math:`F` is the
        frequency axis, and :math:`T` is the time axis.
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

    :returns:
        Augmented spectrogram. *Shape:* Same as input.
    """

    specgram = specgram.clone()

    if specgram.dim() == 2:
        specgram = specgram.unsqueeze(0)

    specgram = warp_along_axis(specgram, warp_axis, max_warp_length)
    specgram = mask_along_axis(
        specgram, 1, num_freq_mask, freq_max_mask_length, freq_mask_max_proportion, mask_value
    )
    specgram = mask_along_axis(
        specgram, 2, num_time_mask, time_max_mask_length, time_mask_max_proportion, mask_value
    )
    return specgram