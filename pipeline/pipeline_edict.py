from typing import Callable, List, Optional, Union

import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from scheduler.scheduling_edict import EDICTScheduler
from utils import auth_token, preprocess

# Getting our HF Auth token
with open("hf_auth", "r") as f:
    auth_token = f.readlines()[0].strip()

clip_filepath = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_filepath)
text_encoder = CLIPTextModel.from_pretrained(clip_filepath, torch_dtype=torch.float16)


dm_filepath = "CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained(
    dm_filepath,
    subfolder="unet",
    use_auth_token=auth_token,
    revision="fp16",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    dm_filepath,
    subfolder="vae",
    use_auth_token=auth_token,
    revision="fp16",
    torch_dtype=torch.float16,
)


class Pipeline:
    def __init__(self) -> None:
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
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

            max_length = text_emb.shape[-1]
            uncond_emb = self._encode(uncond_tokens, max_length, device)

            text_emb = torch.cat([uncond_emb, text_emb])

        return text_emb

    def prepare_latents(self, image, text_emb, device, generator, dtype):
        image = image.to(device=device, dtype=dtype)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        init_latents = 0.18215 * init_latents

        model_input, base = init_latents, init_latents.clone()

        for t in self.scheduler.timesteps:
            if self.scheduler.do_mixing_now:
                base, model_input = self.scheduler.inverse_mixing_step(
                    base, model_input
                )

            # predict the noise residual
            noise_pred = self.unet(
                model_input, t, encoder_hidden_states=text_emb
            ).sample

            base, model_input = self.scheduler.noise_step(
                base, model_input, noise_pred, t
            )

        return base, model_input


    def __call__(
            self,
            image: Union[torch.Tensor, Image.Image],
            prompt_base: str,
            prompt_target: str,
            strength: float,
            negative_prompt: str,
            device: str,
            num_inference_steps: int,
            guidance_scale: float,
            generator,
            dtype,
        ):
            do_classifier_free_guidance = True

            base_emb = self._encode_prompt(
                prompt_base, negative_prompt, device, do_classifier_free_guidance=True
            )

            target_emb = self._encode_prompt(
                prompt_target, negative_prompt, device, do_classifier_free_guidance=False
            )

            image = preprocess(image)

            num_effective_steps = strength * num_inference_steps
            self.scheduler.set_timesteps(num_effective_steps, device=device)

            # do noising steps
            # last iteration returns: y_t, x_t
            model_input, base = self.prepare_latents(
                image, base_emb, device, generator, dtype
            )
