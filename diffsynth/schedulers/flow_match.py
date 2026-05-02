import torch, math



class FlowMatchScheduler():

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None, exponential_shift_mu=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def _match_timestep_indices(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep_tensor = timestep.to(self.timesteps.device).reshape(-1)
        else:
            timestep_tensor = torch.tensor([timestep], device=self.timesteps.device, dtype=self.timesteps.dtype)
        diffs = (self.timesteps[:, None] - timestep_tensor[None, :]).abs()
        timestep_ids = torch.argmin(diffs, dim=0)
        return timestep_ids


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        timestep_ids = self._match_timestep_indices(timestep)
        sigma = self.sigmas[timestep_ids].to(device=sample.device, dtype=sample.dtype)

        if to_final:
            sigma_next = torch.ones_like(sigma) if (self.inverse_timesteps or self.reverse_sigmas) else torch.zeros_like(sigma)
        else:
            next_ids = torch.clamp(timestep_ids + 1, max=len(self.timesteps) - 1)
            sigma_next = self.sigmas[next_ids].to(device=sample.device, dtype=sample.dtype)
            last_mask = (timestep_ids + 1) >= len(self.timesteps)
            if last_mask.any():
                terminal = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
                sigma_next[last_mask] = terminal

        delta = (sigma_next - sigma).view(-1, *([1] * (sample.ndim - 1)))
        prev_sample = sample + model_output * delta
        return prev_sample


    def return_to_timestep(self, timestep, sample, sample_stablized):
        timestep_ids = self._match_timestep_indices(timestep)
        sigma = self.sigmas[timestep_ids].to(device=sample.device, dtype=sample.dtype)
        sigma = sigma.view(-1, *([1] * (sample.ndim - 1)))
        model_output = (sample - sample_stablized) / sigma
        return model_output


    def add_noise(self, original_samples, noise, timestep):
        timestep_ids = self._match_timestep_indices(timestep)
        sigma = self.sigmas[timestep_ids].to(device=original_samples.device, dtype=original_samples.dtype)
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample


    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target


    def training_weight(self, timestep):
        timestep_ids = self._match_timestep_indices(timestep)
        weights = self.linear_timesteps_weights[timestep_ids]
        return weights.mean() if weights.numel() > 1 else weights.squeeze(0)
    
    
    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
