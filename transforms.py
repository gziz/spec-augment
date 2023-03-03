import torch
import functional as F

class SpecAugment(torch.nn.Module):
    """
    Data augmentation for spectrogram (audio)
    Args
        warp_axis: Axis where the warp takes place (0->freq, 1->time)
        warp_param: Boundaries where warp takes place (W, N - W)
        freq_mask_num: Number of masks to apply to the frequency axis
        freq_mask_param: Max length of any individual frequency mask
        freq_mask_p: Max proportion that any individual freq mask can have
        time_mask_num: Number of masks to apply to the time axis
        time_mask_param: Max length of any individual time mask
        time_mask_p: Max proportion that any individual time mask can have
    """
    def __init__(
        self,
        warp_axis: int, 
        warp_param: int = 0,
        freq_mask_num: int = 0,
        freq_mask_param: int = 0,
        freq_mask_p: float = 0.0,
        time_mask_num: int = 0,
        time_mask_param: int = 0,
        time_mask_p: float = 0.0,
        mask_value: float = 0.0

    ) -> None:
        super(SpecAugment, self).__init__()
        self.warp_axis = warp_axis
        self.warp_param = warp_param
        self.freq_mask_num = freq_mask_num
        self.freq_mask_param = freq_mask_param
        self.freq_mask_p = freq_mask_p
        self.time_mask_num = time_mask_num
        self.time_mask_param = time_mask_param
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value

        
    def forward(self, specgram: torch.Tensor):
        """
        Args 
            specgram: Tensor with dimensions (batch, frequency, time)
        Returns
            specgram: Augmented specgram tensor with dimensions (batch, frequency, time)
        """
        if specgram.dim() == 2:
            specgram = specgram.unsqueeze_(0)

        specgram = F.warp_axis_torch(specgram, self.warp_axis, self.warp_param)
        specgram = F.mask_along_axis(specgram, 1, self.freq_mask_num, self.freq_mask_param, self.freq_mask_p, self.mask_value)
        specgram = F.mask_along_axis(specgram, 2, self.time_mask_num, self.time_mask_param, self.time_mask_p, self.mask_value)
        return specgram
