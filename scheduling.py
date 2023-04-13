
import numpy as np
import torch


class EDICTScheduler:
    def __init__(
            self,
            p=0.93,
            beta_1=0.00085,
            beta_T=0.012,
            num_train_timesteps=1000,  # T = 1000
            set_alpha_to_one=False,
            
    ) -> None:
        self.p = p
        self.num_train_timesteps = num_train_timesteps

        # scaled linear
        betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # For PEP 412's sake
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def set_timesteps(self, num_inference_steps, device):

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def denoise_mixing_layer(self, x, y):
        x = self.p * x + (1 - self.p) * y
        y = self.p * y + (1 - self.p) * x

        return [x, y]

    def noise_mixing_layer(self, x, y):
        y = (y - (1 - self.p) * x) / self.p
        x = (x - (1 - self.p) * y) / self.p

        return [x, y]

    def get_alpha_and_beta(self, t):
        # as self.alphas_cumprod is always in cpu
        t = int(t)

        alpha_prod = self.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod

        return alpha_prod, 1 - alpha_prod
    
    def noise_step(
        self,
        base,
        model_input,
        model_output,
        timestep: int,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t ** 0.5) + beta_prod_t_prev ** 0.5

        next_model_input = (base - b_t * model_output) / a_t

        return model_input, next_model_input.to(base.dtype)
    
    def denoise_step(
        self,
        base,
        model_input,
        model_output,
        timestep,
    ):
        prev_timestep = timestep - self.num_train_timesteps / self.num_inference_steps

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, beta_prod_t_prev = self.get_alpha_and_beta(prev_timestep)

        a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        b_t = -a_t * (beta_prod_t ** 0.5) + beta_prod_t_prev ** 0.5
        next_model_input = a_t * base + b_t * model_output

        return model_input, next_model_input.to(base.dtype)