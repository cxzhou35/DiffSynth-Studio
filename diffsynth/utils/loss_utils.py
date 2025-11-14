# adapted from https://github.com/NJU-PCALab/STAR/blob/main/video_super_resolution/scripts/train_sr.py
import torch
import torch.fft
import torch.nn.functional as F

def fourier_transform(x, balance=None):
    """
    Apply Fourier transform to the input tensor and separate it into low-frequency and high-frequency components.

    Args:
    x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
    balance (torch.Tensor or float, optional): Learnable balance parameter for adjusting the cutoff frequency.

    Returns:
    low_freq (torch.Tensor): Low-frequency components (with real and imaginary parts)
    high_freq (torch.Tensor): High-frequency components (with real and imaginary parts)
    """
    # Perform 2D Real Fourier transform (rfft2 only computes positive frequencies)
    x = x.to(torch.float32)
    fft_x = torch.fft.rfft2(x, dim=(-2, -1))

    # Calculate magnitude of frequency components
    magnitude = torch.abs(fft_x)

    # Set cutoff based on balance or default to the 80th percentile of the magnitude for low frequency
    if balance is None:
        # Downsample the magnitude to reduce computation for large tensors
        subsample_size = 10000  # Adjust based on available memory and tensor size
        if magnitude.numel() > subsample_size:
            # Randomly select a subset of values to approximate the quantile
            magnitude_sample = magnitude.flatten()[torch.randint(0, magnitude.numel(), (subsample_size,))]
            cutoff = torch.quantile(magnitude_sample, 0.8)  # 80th percentile for low frequency
        else:
            cutoff = torch.quantile(magnitude, 0.8)  # 80th percentile for low frequency
    else:
        # balance is clamped for safety and used to scale the mean-based cutoff
        cutoff = magnitude.mean() * (1 + 10 * balance)

    # Smooth mask using sigmoid to ensure gradients can pass through
    sharpness = 10  # A parameter to control the sharpness of the transition
    low_freq_mask = torch.sigmoid(sharpness * (cutoff - magnitude))

    # High-frequency mask can be derived from low-frequency mask (1 - low_freq_mask)
    high_freq_mask = 1 - low_freq_mask

    # Separate low and high frequencies using smooth masks
    low_freq = fft_x * low_freq_mask
    high_freq = fft_x * high_freq_mask

    # Return real and imaginary parts separately
    low_freq = torch.stack([low_freq.real, low_freq.imag], dim=-1)
    high_freq = torch.stack([high_freq.real, high_freq.imag], dim=-1)

    return low_freq, high_freq


def extract_frequencies(video: torch.Tensor, balance=None):
    """
    Extract high-frequency and low-frequency components of a video using Fourier transform.

    Args:
    video (torch.Tensor): Input video tensor of shape [batch_size, channels, height, width]

    Returns:
    low_freq (torch.Tensor): Low-frequency components of the video
    high_freq (torch.Tensor): High-frequency components of the video
    """

    # Apply Fourier transform to each frame
    low_freq, high_freq = fourier_transform(video, balance=balance)

    return low_freq, high_freq


class Freq_loss(torch.nn.Module):
    def __init__(self, balance, time_length) -> None:
        super(Freq_loss, self).__init__()
        self.balance = balance
        self.time_length = time_length


    def forward(self, x_pred: torch.Tensor, x_gt: torch.Tensor, timestep: torch.Tensor):
        """
        x_pred: (B, C, H, W)
        x_gt: (B, C, H, W)
        """
        low_freq_x_pred, high_freq_x_pred = extract_frequencies(x_pred, balance=self.balance)
        low_freq_x_gt, high_freq_x_gt= extract_frequencies(x_gt, balance=self.balance)

        loss_low = F.l1_loss(low_freq_x_pred.float(), low_freq_x_gt.float(), reduction="mean")
        loss_high = F.l1_loss(high_freq_x_pred.float(), high_freq_x_gt.float(), reduction="mean")
        ct = (timestep / self.time_length) ** 2

        freq_loss = ct * loss_low + (1 - ct) * loss_high

        return freq_loss
