import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.transforms import InterpolationMode
import re
import random
import torch.nn.functional as F
import torch.nn as nn

vae_path = "log/test_vae/checkpoint-4"

image_folder = "ISTD+_Dataset/test"
image_dir = os.path.join(image_folder, "origin")
latent_dir = os.path.join(image_folder, "latents_sample")

result_dir = "log/result"
image_filenames = sorted(os.listdir(image_dir))
latent_filenames = sorted(os.listdir(latent_dir))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed value
seed = 1234  # You can change this to any integer value
set_seed(seed)

add_dim = False
add_dino = False

weight_dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained(
    vae_path, 
    torch_dtype=weight_dtype,
    subfolder="vae"
)
vae.to(device)

add_dim = vae.config.add_dim
add_dino = vae.config.add_dino

if add_dino:
    DINO_Net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(torch.float16).to(device)

def extract_number(filename):
    match = re.search(r'\d+',filename)
    return int(match.group()) if match else -1

def image_transform(image):
    image = pil_to_tensor(image)
    height, width = image.shape[-2:]

    img_multiple_of = 16
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    image = F.pad(image, (0, padw, 0, padh), 'reflect')

    new_height = height + padh
    new_width = width + padw
    
    image: torch.Tensor = image / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
    image = image.unsqueeze(0)
    image = image.to(weight_dtype).to(device)
    return image, height, width, new_height, new_width

def latent_transform(latent):
    latent = torch.from_numpy(latent)
    latent = latent.unsqueeze(0)
    latent = latent.to(weight_dtype).to(device)

    return latent

def save_image(image_tensor, save_path):
    image = (image_tensor.squeeze() / 2 + 0.5 ).clamp(0, 1) * 255
    image = image.permute(1, 2, 0).cpu().detach()
    image = image.numpy().round().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path)

with torch.no_grad():
    for i in tqdm(range(len(image_filenames))):
        latent_enc_features = None
        dino = None
        file_name = image_filenames[i]
        condition_image_path = os.path.join(image_dir, file_name)
        latent_path = os.path.join(latent_dir, latent_filenames[i])
        condition_image = Image.open(condition_image_path).convert("RGB")
        condition_image, height, width, new_height, new_width = image_transform(condition_image)
        latent = np.load(latent_path)
        latent = latent_transform(latent)

        latent = latent / vae.config.scaling_factor
        if add_dim:
            latent_enc_features = vae.encode(condition_image).enc_feature_list

            if add_dino:
                    DINO_patch_size = 14
                    UpSample = nn.UpsamplingBilinear2d(
                        size=((int)(new_height * (DINO_patch_size / 16)), 
                            (int)(new_width * (DINO_patch_size / 16))))
                    input_dino = UpSample(condition_image).to(torch.float16).to(device)
                    dino = DINO_Net.get_intermediate_layers(input_dino, 4, True)[3]

        image = vae.decode(latent, enc_features=latent_enc_features, dino=dino, return_dict=False)[0]

        
        result_path = os.path.join(result_dir, file_name)
        save_image(image, result_path)
        