import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

model_path = "stabilityai/stable-diffusion-2"
# vae_path = "stabilityai/stable-diffusion-2"  #for stage one inference

unet_path = "log/test/checkpoint-4/unet"
vae_path = "log/test_vae/checkpoint-4"

image_folder = "ISTD+_Dataset/test/"
image_dir = os.path.join(image_folder, "origin")
result_dir = "log/result"
image_filenames = sorted(os.listdir(image_dir))
prompt = ""

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

device = 'cuda'
add_ssao = False
add_depth = False
prediction_type = "sample"
start_from_xt = False
num_train_timesteps = 1000
output_type = 'pil'  # or latent for stage two
# output_type = 'latent'
add_dim = False
add_dino = False
infer_torch_dtype = torch.float16

unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=infer_torch_dtype)
vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=infer_torch_dtype,
    subfolder="vae"
)
add_dim = vae.config.add_dim
add_dino = vae.config.add_dino
if add_dino:
    DINO_Net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(infer_torch_dtype).to(device)
    # DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local').to(infer_torch_dtype).to(device)

noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
if prediction_type is not None:
    # set prediction_type of scheduler if defined
    noise_scheduler.register_to_config(prediction_type=prediction_type)
if start_from_xt and num_train_timesteps is not None:
    noise_scheduler.register_to_config(num_train_timesteps=num_train_timesteps)
pipe = StableDiffusionPipeline.from_pretrained(model_path, unet=unet, vae=vae, scheduler=noise_scheduler, torch_dtype=infer_torch_dtype)
pipe.to("cuda")

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
    return image, height, width, new_height, new_width


with torch.no_grad():
    for i in tqdm(range(len(image_filenames))):
            dino = None
            file_name = image_filenames[i]
            condition_image_path = os.path.join(image_dir, file_name)
            condition_image = Image.open(condition_image_path).convert("RGB")
            condition_image, height, width, new_height, new_width = image_transform(condition_image)

            if add_dino:
                DINO_patch_size = 14
                UpSample = nn.UpsamplingBilinear2d(
                    size=((int)(new_height * (DINO_patch_size / 16)), 
                        (int)(new_width * (DINO_patch_size / 16))))
                input_dino = UpSample(condition_image).to(infer_torch_dtype).to(device)
                dino = DINO_Net.get_intermediate_layers(input_dino, 4, True)[3]
            
            image = pipe(condition_image, prompt, dino = dino, add_dim=add_dim, num_inference_steps=20, output_type=output_type).images[0]

            if output_type == 'pil':
                image = image.crop((0, 0, width, height))
                result_path = os.path.join(result_dir, file_name)
                image.save(result_path)
            else:
                result_path = os.path.join(result_dir, 'latent_' + file_name.split(".")[0] + '.npy')
                np.save(result_path, image.cpu())