from typing import Callable, List, Optional, Union

import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPModel, CLIPTokenizer

from scheduler.scheduling_edict import EDICTScheduler
from utils import preprocess

# Getting our HF Auth token
with open("hf_auth", "r") as f:
    auth_token = f.readlines()[0].strip()

clip_filepath = "openai/clip-vit-large-patch14"


dm_filepath = "CompVis/stable-diffusion-v1-4"


class Pipeline:
    def __init__(self) -> None:
        
        device = "cuda"

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_filepath)
        self.text_encoder = CLIPModel.from_pretrained(
            clip_filepath, torch_dtype=torch.float16
        ).text_model.to(device)

        self.unet = UNet2DConditionModel.from_pretrained(
            dm_filepath,
            subfolder="unet",
            use_auth_token=auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to(device)

        self.vae = AutoencoderKL.from_pretrained(
            dm_filepath,
            subfolder="vae",
            use_auth_token=auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to(device)

        self.scheduler = EDICTScheduler()

    def _encode(self, prompt, max_length, device):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        attention_mask = None

        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask
        )[0]

        return text_embeddings

    def _encode_prompt(
        self, prompt, negative_prompt, device, do_classifier_free_guidance
    ):
        text_emb = self._encode(prompt, self.tokenizer.model_max_length, device)

        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            max_length = text_emb.shape[1]
            uncond_emb = self._encode(uncond_tokens, max_length, device)

            text_emb = torch.cat([uncond_emb, text_emb])

        return text_emb

    @torch.no_grad()
    def prepare_latents(self, image, text_emb, do_classifier_free_guidance, guidance_scale, device, generator, dtype):
        image = image.to(device=device, dtype=self.vae.dtype)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = 0.18215 * init_latents

        model_input, base = init_latents, init_latents.clone()

        for i, t in enumerate(self.scheduler.fwd_timesteps):
            if self.scheduler.do_mixing_now:
                base, model_input = self.scheduler.inverse_mixing_step(
                    base, model_input
                )

            latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_emb
            ).sample

            if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            base, model_input = self.scheduler.noise_step(
                base, model_input, noise_pred, t
            )

        return base, model_input

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        base_prompt: str,
        target_prompt: str,
        strength: float,
        negative_prompt: str = None,
        device: str = 'cuda',
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        dtype=torch.float64,
    ):
        
        generator = torch.cuda.manual_seed(42.0)

        do_classifier_free_guidance = True

        base_emb = self._encode_prompt(
            base_prompt, negative_prompt, device, do_classifier_free_guidance
        )

        target_emb = self._encode_prompt(
            target_prompt, negative_prompt, device, do_classifier_free_guidance
        )

        image = preprocess(image)

        num_effective_steps = int(strength * num_inference_steps)
        self.scheduler.set_timesteps(num_effective_steps, device=device)

        # do noising steps
        # last iteration returns: y_t, x_t
        model_input, base = self.prepare_latents(
            image, base_emb, do_classifier_free_guidance, guidance_scale, device, generator, dtype
        )

        bwd_timesteps = self.scheduler.fwd_timesteps.flip(0)
        self.scheduler.do_mixing_now = False

        for i, t in enumerate(bwd_timesteps):

            latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=target_emb).sample

            if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            base, model_input = self.scheduler.denoise_step(base, model_input, noise_pred, t)
        
        

       
