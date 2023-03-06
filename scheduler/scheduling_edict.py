from typing import Union

import numpy as np
import torch


class EDICTScheduler:
    def __init__(
        self,
        beta_1=0.00085,
        beta_T=0.012,
        p=0.75,
        eta=0.0,
        num_train_timesteps=1000,
    ) -> None:
        self.eta = eta
        self.p = p

        betas = (
            torch.linspace(
                beta_1**0.5, beta_T**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.sqrt_betas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = None
        self.step_ratio = None
        self.do_mixing_now = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        self.num_inference_steps = num_inference_steps
        self.step_ratio = self.num_train_timesteps // self.num_inference_steps

        timesteps = (
            (np.arange(0, num_inference_steps) * self.step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).repeat_interleave(2)
        self.timesteps = self.timesteps.to(device)

        self.do_mixing_now = True

    # Start with self.do_mixing_now = False
    def denoise_step(self, base, model_input, model_output, t):
        # no skipped timesteps
        t_prev = t - self.step_ratio
        a_t = self.sqrt_alphas_cumprod[t_prev] / self.sqrt_alphas_cumprod[t]
        b_t = -a_t * self.sqrt_betas_cumprod[t] + self.sqrt_betas_cumprod[t_prev]

        next_model_input = a_t * base + b_t * model_output

        if not self.do_mixing_now:
            # It implies we just did equation (14.1) --> next_model_input = x_t_inter
            self.do_mixing_now ^= True
            return model_input, next_model_input
        else:
            # It implies we just did equation (14.2) --> next_model_input = y_t_inter
            # Do equation (14.3)
            self.do_mixing_now ^= True
            return self._forward_mixing_step(model_input, next_model_input)

    def _forward_mixing_step(self, x, y):
        x_prev = self.p * x + (1 - self.p) * y
        y_prev = self.p * y + (1 - self.p) * x_prev

        return x_prev, y_prev

    # Start with self.do_mixing_now = True
    def noise_step(self, base, model_input, model_output, t):
        t_prev = t - self.step_ratio
        a_t = self.sqrt_alphas_cumprod[t_prev] / self.sqrt_alphas_cumprod[t]
        b_t = -a_t * self.sqrt_betas_cumprod[t] + self.sqrt_betas_cumprod[t_prev]

        next_model_input = (base - b_t * model_output) / a_t

        self.do_mixing_now ^= True

        return model_input, next_model_input

    def inverse_mixing_step(self, base, model_input):
        base = (base - (1 - self.p) * model_input) / self.p
        model_input = (model_input - (1 - self.p) * base) / self.p

        return base, model_input
