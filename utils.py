import torch
from PIL import Image
import numpy as np

def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [
            np.array(i.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
        
    return image