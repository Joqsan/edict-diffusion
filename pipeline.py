import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from scheduling import EDICTScheduler

model_path_clip = "openai/clip-vit-large-patch14"
model_path_diffusion = "CompVis/stable-diffusion-v1-4"


def preprocess(image):
    if isinstance(image, Image.Image):
        w, h = image.size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    else:
        raise TypeError("Expected object of type PIL.Image.Image")
    return image


class Pipeline:
    def __init__(self, leapfrog_steps=True, device="cuda") -> None:
        
        self.leapfrog_steps = leapfrog_steps
        self.device = device

        self.scheduler = EDICTScheduler(
            p=0.93,
            beta_1=0.00085,
            beta_T=0.012,
            num_train_timesteps=1000
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path_diffusion,
            subfolder="unet",
            revision="fp16",
            torch_dtype=torch.float16,
        ).double().to(device)
        
        self.vae = AutoencoderKL.from_pretrained(
            model_path_diffusion,
            subfolder="vae",
            revision="fp16",
            torch_dtype=torch.float16,
        ).double().to(device)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
        self.encoder = CLIPTextModel.from_pretrained(
            model_path_clip,
            torch_dtype=torch.float16
        ).double().to(device)


    def encode_prompt(self, prompt, negative_prompt=None):
        null_prompt = "" if negative_prompt is None else negative_prompt

        tokens_uncond = self.tokenizer(
            null_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embeds_uncond = self.encoder(
            tokens_uncond.input_ids.to(self.device)
        ).last_hidden_state

        tokens_cond = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        embeds_cond = self.encoder(
            tokens_cond.input_ids.to(self.device)
        ).last_hidden_state

        return torch.cat([embeds_uncond, embeds_cond])

    @torch.no_grad()
    def decode_latents(self, latents):
        # latents = 1 / self.vae.config.scaling_factor * latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.no_grad()
    def prepare_latents(self, image, text_embeds, timesteps, guidance_scale):
        generator = torch.cuda.manual_seed(1)
        image = image.to(device=self.device, dtype=text_embeds.dtype)
        latent = self.vae.encode(image).latent_dist.sample(generator)

        # init_latents = self.vae.config.scaling_factor * init_latents
        latent = 0.18215 * latent

        latent_pair = [latent.clone(), latent.clone()]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            latent_pair = self.scheduler.noise_mixing_layer(x=latent_pair[0], y=latent_pair[1])

            # j - model_input index, k - base index
            for j in range(2):
                k = j ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 0:
                        k, j = j, k

                model_input = latent_pair[j]
                base = latent_pair[k]

                latent_model_input = torch.cat([model_input] * 2)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                base, model_input = self.scheduler.noise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )
            
                latent_pair[k] = model_input

        return latent_pair

    

    @torch.no_grad()
    def __call__(
        self,
        base_prompt,
        target_prompt,
        image,
        guidance_scale=3.0,
        steps=50,
        strength=0.8,
    ):
        image = preprocess(image)  # from PIL.Image to torch.Tensor

        base_embeds = self.encode_prompt(base_prompt)
        target_embeds = self.encode_prompt(target_prompt)

        self.scheduler.set_timesteps(steps, self.device)

        t_limit = steps - int(steps * strength)
        fwd_timesteps = self.scheduler.timesteps[t_limit:]
        bwd_timesteps = fwd_timesteps.flip(0)

        latent_pair = self.prepare_latents(image, base_embeds, bwd_timesteps, guidance_scale)

        for i, t in tqdm(enumerate(fwd_timesteps), total=len(fwd_timesteps)):

            # j - model_input index, k - base index
            for k in range(2):
                j = k ^ 1

                if self.leapfrog_steps:
                    if i % 2 == 1:
                        k, j = j, k

                model_input = latent_pair[j]
                base = latent_pair[k]

                latent_model_input = torch.cat([model_input] * 2)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=target_embeds).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                base, model_input = self.scheduler.denoise_step(
                    base=base,
                    model_input=model_input,
                    model_output=noise_pred,
                    timestep=t,
                )

                latent_pair[k] = model_input

            latent_pair = self.scheduler.denoise_mixing_layer(x=latent_pair[0], y=latent_pair[1])

        # either one is fine
        final_latent = latent_pair[0]

        image = self.decode_latents(final_latent)
        image = (image[0] * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)

        return pil_image