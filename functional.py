import numpy as np
import torch
from torch import Tensor


def warp_axis_torch(specgram:Tensor, axis:int, W:float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Axis where the warp takes place
        W: Boundary of time steps where the warp takes place (W, num_warped_axis - W)
    Returns:
        Tensor: Warped spectrogram of dimensions (batch, freq, time)
    """
    
    if axis not in [1, 2]:
            raise ValueError("Only Frequency and Time masking are supported")
            
    num_warped = specgram.shape[axis]
    num_non_warped = specgram.shape[1 if axis == 2 else 2]

    if W == 0:
        return specgram
    assert 2 * W < num_warped, (
        f"W param {W} must be smaller than half the size of the warped axis {num_warped}")
    
    w0 = torch.randint(W, num_warped - W, ())
    w = torch.randint(-W + 1, W, ())

    if axis == 1:
        lower, upper = specgram[:, :w0, :], specgram[:, w0:, :]
        lower_sz, upper_sz = (w0 + w, num_non_warped), (num_warped - w0 - w, num_non_warped)
    else:
        lower, upper = specgram[:, :, :w0], specgram[:, :, w0:]
        lower_sz, upper_sz = (num_non_warped, w0 + w), (num_non_warped, num_warped - w0 - w)
    
    # interpolate receives 4D: (batch, channel, freq, time)
    lower = lower.unsqueeze(1)
    upper = upper.unsqueeze(1)

    lower = torch.nn.functional.interpolate(lower, size=lower_sz, mode='bilinear')
    upper = torch.nn.functional.interpolate(upper, size=upper_sz, mode='bilinear')
    
    lower.squeeze_(1)
    upper.squeeze_(1)

    specgram = torch.cat((lower, upper), axis=axis)
    return specgram



def mask_along_axis(
    specgram: Tensor,
    axis: int,
    num_masks: int,
    mask_param: int,
    p: float = 0.0,
    mask_value: float = 0.0
):
    """
    Apply mask along a spectrogram.
    The length of the mask is randomly chosen, with a cap on mask_param.
    Args
        specgram: A tensor with dimensions (batch, freq, time)
        axis: Masking is applied (freq -> 1, time -> 2)
        num_masks: Number of masks
        mask_param: Max length allowed for each individual mask.
        p: Max proportion of masked rows/cols for each individual mask.
        mask_value: Value for the masked portions
    Returns
        Tensor: Masked spectrogram of dimensions (batch, freq, time)
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    mask_param = min(mask_param, int(specgram.shape[axis] * p))
    if mask_param < 1:
        return specgram
    
    value = torch.randint(mask_param, ())

    for _ in range(num_masks):
        mask_start = torch.randint(specgram.shape[axis] - value, ())
        mask_end = mask_start + value

        if axis == 1:
            specgram[:, mask_start: mask_end, :] = mask_value
        else:
            specgram[:, :, mask_start: mask_end] = mask_value

    return specgram



def spec_augment(
    specgram: Tensor,
    warp_axis: int, 
    warp_param: int = 0,
    freq_mask_num: int = 0,
    freq_mask_param: int = 0,
    freq_mask_p: float = 0.0,
    time_mask_num: int = 0,
    time_mask_param: int = 0,
    time_mask_p: float = 0.0,
    mask_value: float = 0.0
):
    """
    SpecAugment to spectrogram with dimensions (batch, frequency, time)
    Args
        specgram: Tensor with dimensions (batch, frequency, time)
        warp_axis: Axis where the warp takes place (0->freq, 1->time)
        warp_param: Boundaries where warp takes place (W, N - W)
        freq_mask_num: Number of masks to apply to the frequency axis
        freq_mask_param: Max length of any individual frequency mask
        freq_mask_p: Max proportion that any individual freq mask can have
        time_mask_num: Number of masks to apply to the time axis
        time_mask_param: Max length of any individual time mask
        time_mask_p: Max proportion that any individual time mask can have
    Returns
        Tensor: Augmented spectrogram with dimensions (batch, frequency, time)
    """
    if specgram.dim() == 2:
        specgram = specgram.unsqueeze_(0)

    specgram = warp_axis_torch(specgram, warp_axis, warp_param)
    specgram = mask_along_axis(specgram, 1, freq_mask_num, freq_mask_param, freq_mask_p, mask_value)
    specgram = mask_along_axis(specgram, 2, time_mask_num, time_mask_param, time_mask_p, mask_value)
    return specgram



# Outdated
def warp_axis_cv2(specgram, axis:int, W:float):
    """
    Warp axis (frequency or time) between W boundaries, starting from point w0, the warp 
    direction can be negative or positive, depending on the randomly chosen distance w.
    Args:
        specgram: A tensor|array with dimensions (freq, time)
        axis: Axis where the warp is applied (freq -> 1, time -> 2)
        W: Boundary of time steps where the warp takes place (W, num_warped_axis - W)
    Returns:
        tensor | array: Warped spectrogram
    """
    num_warped = specgram.shape[axis]
    num_non_warped = specgram.shape[abs(axis-1)]

    if W == 0:
        return specgram
    assert 2 * W < num_warped, (
        f"W param {W} must be smaller than half the size of the warped axis {num_warped}")
    
    w0 = np.random.randint(W, num_warped - W)
    w = np.random.randint(-W + 1, W)

    if axis == 0:
        lower, upper = specgram[:w0, :], specgram[w0:, :]
        lower_sz, upper_sz = (num_non_warped, w0 + w), (num_non_warped, num_warped - w0 - w)
    else:
        lower, upper = specgram[:, :w0], specgram[:, w0:]
        lower_sz, upper_sz = (w0 + w, num_non_warped), (num_warped - w0 - w, num_non_warped)

    import cv2
    lower = cv2.resize(lower, dsize=lower_sz, interpolation=cv2.INTER_LINEAR)
    upper = cv2.resize(upper, dsize=upper_sz, interpolation=cv2.INTER_LINEAR,)
    specgram = np.concatenate((lower, upper), axis=axis)

    return specgram