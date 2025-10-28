import torch
from .sd_unet import UpSampler, DownSampler, ResnetBlock
from .sd_vae_encoder import VAEAttentionBlock
from .sd_vae_decoder import VAEAttentionBlock
from .flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..utils.ideal_lpf import LPF_RFFT, UpsampleRFFT, WarpedNonlinearity, wrap_nonlinearity, WarpedConvIn


class ALUpSampler(UpSampler):
    """
    Anti-aliased UpSampler
    """
    def __init__(self, channels):
        super().__init__(channels)

        self.up_layer = UpsampleRFFT()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        hidden_states = self.up_layer(hidden_states)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class ALDownSampler(DownSampler):
    """
    Anti-aliased DownSampler
    """
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__(channels, padding=padding, extra_padding=extra_padding)

        # change the conv stride to 1
        self.conv.stride = 1
        self.lpf = LPF_RFFT()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            # TODO: check the padding values, originally is (0, 1, 0, 1)
            pad = (1, 1, 1, 1)
            hidden_states = torch.nn.functional.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.lpf(hidden_states)
        hidden_states = hidden_states[:, :, ::2, ::2]
        return hidden_states, time_emb, text_emb, res_stack


class ALResnetBlock(ResnetBlock):
    """
    Anti-aliased Resnet Block
    """
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__(in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5)

        self.nonlinearity = wrap_nonlinearity(self.nonlinearity)


def wrap_upsampler_with_al(upsampler: UpSampler):
    """
    Wrap an UpSampler with anti-aliased components
    """
    wrapped_upsampler = ALUpSampler(upsampler.channels)

    return wrapped_upsampler


def wrap_downsampler_with_al(downsampler: DownSampler):
    """
    Wrap a DownSampler with anti-aliased components
    """
    wrapped_downsampler = ALDownSampler(
        downsampler.channels,
        padding=downsampler.padding,
        extra_padding=downsampler.extra_padding
    )

    return wrapped_downsampler


def wrap_resnetblock_with_al(resnet_block: ResnetBlock):
    """
    Wrap a ResnetBlock with anti-aliased components
    """
    wrapped_block = ALResnetBlock(
        resnet_block.in_channels,
        resnet_block.out_channels,
        eps=resnet_block.norm1.eps
    )

    return wrapped_block


class FluxALVAEEncoder(FluxVAEEncoder):
    def __init__(self, use_al=False):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 32, kernel_size=3, padding=1)

        # replace the original blocks with anti-aliased components
        if use_al:
            for i, block in enumerate(self.blocks):
                if isinstance(block, ResnetBlock):
                    self.blocks[i] = wrap_resnetblock_with_al(block)
                elif isinstance(block, DownSampler):
                    self.blocks[i] = wrap_downsampler_with_al(block)

            self.conv_act = wrap_nonlinearity(self.conv_act)



class FluxALVAEDecoder(FluxVAEDecoder):
    def __init__(self):
        super().__init__()

        self.conv_in = torch.nn.Conv2d(16, 512, kernel_size=3, padding=1) # Different from SD 1.x
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)

        # replace the original blocks with anti-aliased components
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if isinstance(block, ResnetBlock):
                self.blocks[i] = wrap_resnetblock_with_al(block)
            elif isinstance(block, UpSampler):
                self.blocks[i] = wrap_upsampler_with_al(block)

        self.conv_act = wrap_nonlinearity(self.conv_act)


def wrap_vae_with_al(vae_encoder: FluxVAEEncoder, vae_decoder: FluxVAEDecoder):
    """
    Wrap a VAE with anti-aliased components
    """
    # wrap vae encoder
    for i, block in enumerate(vae_encoder.blocks):
        if isinstance(block, ResnetBlock):
            vae_encoder.blocks[i] = wrap_resnetblock_with_al(block)
        elif isinstance(block, DownSampler):
            vae_encoder.blocks[i] = wrap_downsampler_with_al(block)
    vae_encoder.conv_act = wrap_nonlinearity(vae_encoder.conv_act)

    # wrap vae decoder
    for i, block in enumerate(vae_decoder.blocks):
        if isinstance(block, ResnetBlock):
            vae_decoder.blocks[i] = wrap_resnetblock_with_al(block)
        elif isinstance(block, UpSampler):
            vae_decoder.blocks[i] = wrap_upsampler_with_al(block)
    vae_decoder.conv_act = wrap_nonlinearity(vae_decoder.conv_act)
