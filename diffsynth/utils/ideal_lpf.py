# Adapted from https://github.com/SingleZombie/AFLDM/blob/main/afldm/af_libs/ideal_lpf.py

import torch
import torch.nn as nn


def create_lpf_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0
        # N % 4 =0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0
        rect_1d[cutoff_high] = 0

    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d

# support not square inputs
def create_lpf_rect_hw(H, W, cutoff=0.5):
    rects = []
    for dim_size in [H, W]:
        cutoff_low = int((dim_size * cutoff) // 2)
        cutoff_high = int(dim_size - cutoff_low)
        rect = torch.ones(dim_size)
        rect[cutoff_low + 1:cutoff_high] = 0
        if dim_size % 4 == 0:
            # if N is divides by 4, nyquist freq should be 0
            # N % 4 =0 means the downsampeled signal is even
            rect[cutoff_low] = 0
            rect[cutoff_high] = 0
        rects.append(rect)
    rect_2d = rects[0][:, None] * rects[1][None, :]
    return rect_2d

def create_fixed_lpf_rect(N, size):
    rect_1d = torch.ones(N)
    if size < N:
        cutoff_low = size // 2
        cutoff_high = int(N - cutoff_low)
        rect_1d[cutoff_low + 1:cutoff_high] = 0
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d

def create_fixed_lpf_rect_hw(H, W, size):
    rects = []
    for dim_size in [H, W]:
        rect = torch.ones(dim_size)
        if size < dim_size:
            cutoff_low = size // 2
            cutoff_high = int(dim_size - cutoff_low)
            rect[cutoff_low + 1:cutoff_high] = 0
        rects.append(rect)
    rect_2d = rects[0][:, None] * rects[1][None, :]
    return rect_2d

# upsample using FFT
def create_recon_rect(N, cutoff=0.5):
    cutoff_low = int((N * cutoff) // 2)
    cutoff_high = int(N - cutoff_low)
    rect_1d = torch.ones(N)
    rect_1d[cutoff_low + 1:cutoff_high] = 0
    if N % 4 == 0:
        # if N is divides by 4, nyquist freq should be 0.5
        # N % 4 = 0 means the downsampeled signal is even
        rect_1d[cutoff_low] = 0.5
        rect_1d[cutoff_high] = 0.5
    rect_2d = rect_1d[:, None] * rect_1d[None, :]
    return rect_2d

def create_recon_rect_hw(H, W, cutoff=0.5):
    rects = []
    for dim_size in [H, W]:
        cutoff_low = int((dim_size * cutoff) // 2)
        cutoff_high = int(dim_size - cutoff_low)
        rect = torch.ones(dim_size)
        rect[cutoff_low + 1:cutoff_high] = 0
        if dim_size % 4 == 0:
            # if N is divides by 4, nyquist freq should be 0.5
            # N % 4 = 0 means the downsampeled signal is even
            rect[cutoff_low] = 0.5
            rect[cutoff_high] = 0.5
        rects.append(rect)

    rect_2d = rects[0][:, None] * rects[1][None, :]
    return rect_2d


class LPF_RFFT(nn.Module):
    '''
    saves rect in first use
    '''

    def __init__(self, cutoff=0.5, transform_mode='rfft', fixed_size=None):
        super(LPF_RFFT, self).__init__()
        self.cutoff = cutoff
        self.fixed_size = fixed_size
        assert transform_mode in [
            'fft', 'rfft'], f'transform_mode={transform_mode} is not supported'
        self.rect_dict = {}
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == 'fft' else torch.fft.rfft2
        self.itransform = (lambda x: torch.real(torch.fft.ifft2(
            x))) if transform_mode == 'fft' else torch.fft.irfft2

    def forward(self, x):
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        x_fft = self.transform(x)
        # Build a separable 2D rectangle mask that matches the FFT output shape:
        # (H, W) for fft2, or (H, W/2+1) for rfft2
        B, C, H, W = x.shape
        key = (H, W, self.transform_mode)
        if key in self.rect_dict:
            rect = self.rect_dict[key].to(x.device)
        else:
            rect = create_lpf_rect_hw(H, W, self.cutoff)
            rect = rect[:, :int(W/2+1)] if self.transform_mode == 'rfft' else rect
            rect = rect.to(x.device)
            self.rect_dict[key] = rect
        x_fft *= rect.to(x_fft.dtype)
        out = self.itransform(x_fft, s=(H, W))
        out = out.to(orig_dtype)

        return out


class LPF_RECON_RFFT(nn.Module):
    '''
    saves rect in first use
    '''

    def __init__(self, cutoff=0.5, transform_mode='rfft'):
        super(LPF_RECON_RFFT, self).__init__()
        self.cutoff = cutoff
        assert transform_mode in [
            'fft', 'rfft'], f'mode={transform_mode} is not supported'
        self.rect_dict = {}
        self.transform_mode = transform_mode
        self.transform = torch.fft.fft2 if transform_mode == 'fft' else torch.fft.rfft2
        self.itransform = (lambda x: torch.real(torch.fft.ifft2(
            x))) if transform_mode == 'fft' else torch.fft.irfft2

    def forward(self, x):
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        x_fft = self.transform(x)
        B, C, H, W = x.shape
        key = (H, W, self.transform_mode)
        if key in self.rect_dict:
            rect = self.rect_dict[key].to(x.device)
        else:
            rect = create_recon_rect_hw(H, W, self.cutoff)
            rect = rect[:, :int(W/2+1)] if self.transform_mode == 'rfft' else rect
            rect = rect.to(x.device)
            self.rect_dict[key] = rect
        x_fft *= rect.to(x_fft.dtype)
        out = self.itransform(x_fft, s=(H, W))
        out = out.to(orig_dtype)
        return out


class UpsampleRFFT(nn.Module):
    '''
    input shape is unknown
    '''

    def __init__(self, up=2, transform_mode='rfft', factor=1):
        super(UpsampleRFFT, self).__init__()
        self.up_scale = up
        self.recon_filter = LPF_RECON_RFFT(
            cutoff=1 / up * factor, transform_mode=transform_mode)

    def forward(self, x):
        # pad zeros
        B, C, H, W = x.shape
        x = x.reshape([B, C, H, 1, W, 1])
        x = torch.nn.functional.pad(x, [0, self.up_scale-1, 0, 0, 0, self.up_scale-1])
        x = x.reshape([B, C, H*self.up_scale, W*self.up_scale])
        x = self.recon_filter(x) * (self.up_scale ** 2)
        return x


def subpixel_shift(images, up=2, shift_x=1, shift_y=1, up_method='ideal'):
    '''
    effective fractional shift is (shift_x / up, shift_y / up)
    '''

    assert up_method == 'ideal', 'Only "ideal" interpolation kenrel is supported'
    up_layer = UpsampleRFFT(up=up).to(images.device)
    up_img_batch = up_layer(images)
    # img_batch_1 = up_img_batch[:, :, 1::2, 1::2]
    img_batch_1 = torch.roll(
        up_img_batch, shifts=(-shift_x, -shift_y), dims=(2, 3))[:, :, ::up, ::up]
    return img_batch_1


class WarpedNonlinearity(nn.Module):
    def __init__(self, nonlinearity):
        super().__init__()
        self.up_layer = UpsampleRFFT()
        self.lpf = LPF_RFFT(1/2)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        """
        @x: (B, C, H, W)
        """
        # TODO: check inputs dim
        if x.ndim < 4:
            return self.nonlinearity(x)

        x = self.up_layer(x)
        x = self.nonlinearity(x)
        x = self.lpf(x)
        x = x[:, :, ::2, ::2]

        return x


class WarpedConvIn(nn.Conv2d):
    def __init__(self, conv: nn.Conv2d, wraped_nonlinearity):
        super().__init__(conv.in_channels, conv.out_channels,
                         conv.kernel_size, conv.stride, conv.padding)
        self.weight = conv.weight
        self.bias = conv.bias
        self.nonlinearity = wraped_nonlinearity

    def forward(self, x):
        x = super().forward(x)
        x = self.nonlinearity(x)
        return x


def wrap_nonlinearity(nonlinearity):
    return WarpedNonlinearity(nonlinearity)
