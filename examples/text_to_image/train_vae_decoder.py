#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from PIL import Image
import datetime
from torchvision.transforms.functional import pil_to_tensor, resize
from torchvision.transforms import InterpolationMode
# import matplotlib.pyplot as plt

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers import MarigoldDepthPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file
import json

from utils.losses import CharbonnierLoss
from utils.contperceptual import LPIPSWithDiscriminator


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def myPSNR(tar_img, prd_img):
    prd_img = (prd_img + 1.0) / 2.0
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def tensor_to_pil(image_tensor):
    image_list = []
    for i in range(image_tensor.shape[0]):
        image = (image_tensor[i] / 2 + 0.5 ).clamp(0, 1) * 255
        image = image.permute(1, 2, 0).cpu().detach()
        image = image.numpy().round().astype(np.uint8)
        image = Image.fromarray(image)
        image_list.append(image)
    return image_list

def reshape_super_resolution(input_tensor):
    B, C, H, W = input_tensor.shape

    tensor_1 = input_tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    tensor_2 = tensor_1.reshape(B, H // 2, 2, W // 2, 2, C)
    tensor_3 = tensor_2.permute(0, 1, 3, 2, 4, 5)  # (B, H/3, W/3, 3, 3, C)
    tensor_4 = tensor_3.reshape(B, H // 2, W // 2, 2 * 2 * C)
    output_tensor = tensor_4.permute(0, 3, 1, 2)  # (B, 3*3*C, H/3, W/3)

    return output_tensor
    
def reverse_reshape_super_resolution(input_tensor):
    B, C, H, W = input_tensor.shape

    tensor_1 = input_tensor.permute(0, 2, 3, 1)
    tensor_2 = tensor_1.reshape(B, H, W, 2, 2, C // 4)
    tensor_3 = tensor_2.permute(0, 1, 3, 2, 4, 5) # (B, H/3, W/3, 3, 3, C)
    tensor_4 = tensor_3.reshape(B, H * 2, W * 2, C // 4)
    out_tensor = tensor_4.permute(0, 3, 1, 2)  # (B, C, H, W)

    return out_tensor


def log_validation(val_dataloader, vae, args, accelerator, weight_dtype, epoch, DINO_Net=None):
    logger.info("Running validation... ")
    weight_dtype = torch.float16
    vae = vae.to(accelerator.device)
    if args.add_dino:
        DINO_Net = DINO_Net.to(accelerator.device)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    val_psnr = []
    for i, val_data in enumerate(val_dataloader):
        with torch.no_grad():
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                gt_image = val_data['pixel_values'].to(weight_dtype).to(accelerator.device)
                condition_image = val_data['conditioning_pixel_values'].to(weight_dtype).to(accelerator.device)
                latent = val_data['latent_values'].to(weight_dtype).to(accelerator.device)

                condition_image_in = condition_image
                if args.super_reshape:
                    condition_image_in = reshape_super_resolution(condition_image)
                if args.add_dim:
                    dino = None
                    latent = latent / vae.module.config.scaling_factor
                    condition_features = vae.module.encode(condition_image_in)
                    latent_enc_features = condition_features.enc_feature_list
                    if args.add_dino:
                        DINO_patch_size = 14
                        H, W = condition_image[0].shape[-2:]
                        scale_factor = DINO_patch_size / 16
                        UpSample = nn.UpsamplingBilinear2d(
                            size=((int)(H * scale_factor), 
                                (int)(W * scale_factor)))
                        with torch.no_grad():
                            input_dino = UpSample(condition_image)
                        dino = DINO_Net.get_intermediate_layers(input_dino, 4, True)[3]
                    out_images = vae.module.decode(latent, enc_features=latent_enc_features, dino=dino, return_dict=False)[0]
                else:
                    latent = latent / vae.config.scaling_factor
                    out_images = vae.decode(latent, return_dict=False)[0]

                if args.super_reshape:
                    out_images = reverse_reshape_super_resolution(out_images)

            val_psnr.append(batch_PSNR(gt_image, out_images))

            if i == 0:
                out_images = tensor_to_pil(out_images)
                images.append(out_images[0])


    avg_psnr = sum(val_psnr) / len(val_dataloader)
    logger.info("[Ep %d \t PSNR SIDD Image: %.4f\t ] " % (epoch, avg_psnr))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation_image", np_images, epoch, dataformats="NHWC")
            tracker.writer.add_scalar("validation_psnr_image", avg_psnr, epoch)
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    # del pipeline
    torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        # required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    ),
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--latent_column",
        type=str,
        default="latent",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=10,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_epochs`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_latent",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_epochs`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Batch size (per device) for the val dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10000000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        # required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=3750,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--add_dim",
        action="store_true",
        default=False,
        help="whether to add cfw",
    )
    parser.add_argument(
        "--add_dino",
        action="store_true",
        default=False,
        help="whether to add cfw",
    )
    parser.add_argument(
        "--super_reshape",
        action="store_true",
        default=False,
        help="whether to super_reshape",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=None,
        help="set num_train_timesteps.",
    )
    parser.add_argument(
        "--start_from_xt",
        action="store_true",
        default=False,
        help="start at xt,not xT",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def make_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                trust_remote_code=True,
                cache_dir=None,
                # data_dir=args.data_dir
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    if args.latent_column is None:
        latent_column = column_names[3]
        logger.info(f"depth column defaulting to {latent_column}")
    else:
        latent_column = args.latent_column
        if latent_column not in column_names:
            raise ValueError(
                f"`--depth_column` value '{args.depth_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def latent_transforms(latent_in):
        latent_in = torch.from_numpy(latent_in)
        return latent_in


    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        latent = [np.array(image) for image in examples[latent_column]]
        latent = [latent_transforms(image) for image in latent]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["latent_values"] = latent

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset

def make_val_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                trust_remote_code=True,
                cache_dir=None,
                # data_dir=args.data_dir
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["test"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    if args.latent_column is None:
        latent_column = column_names[3]
        logger.info(f"depth column defaulting to {latent_column}")
    else:
        latent_column = args.latent_column
        if latent_column not in column_names:
            raise ValueError(
                f"`--depth_column` value '{args.depth_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def conditioning_image_transforms(image):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = transform(image)
        height, width = image.shape[-2:]
        img_multiple_of = 16
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
            (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        image_pad = F.pad(image, (0, padw, 0, padh), 'reflect')

        return image_pad

    def latent_transforms(latent_in):
        latent_in = torch.from_numpy(latent_in)

        return latent_in

    def preprocess_test(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        latent = [np.array(image) for image in examples[latent_column]]
        latent = [latent_transforms(image) for image in latent]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["latent_values"] = latent
        return examples

    with accelerator.main_process_first():
        if args.max_test_samples is not None:
            dataset["test"] = dataset["test"].select(range(args.max_test_samples))\
        # Set the testing transforms
        test_dataset = dataset["test"].with_transform(preprocess_test)

    return test_dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    latent_values = torch.stack([example["latent_values"] for example in examples])
    latent_values = latent_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "latent_values": latent_values
    }

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # loss
    criterion_restore = CharbonnierLoss().to(accelerator.device)
    dim_loss = LPIPSWithDiscriminator(
        disc_start=501,
        kl_weight=0,
        disc_weight=0.025,
        disc_factor=1.0).to(accelerator.device)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

        # logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log_file_path = os.path.join(args.output_dir, datetime.datetime.now().isoformat(timespec='minutes')+'.txt')

        file_handler = logging.FileHandler(log_file_path)
        # file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.logger.addHandler(file_handler)

    vae_ori = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    config_path = "./vae/config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    if args.add_dim:
        config_dict["up_block_types"] = [
            "UpDecoderBlock2DCfw",
            "UpDecoderBlock2DCfw",
            "UpDecoderBlock2DCfw",
            "UpDecoderBlock2DCfw"
        ]
    vae = AutoencoderKL(**config_dict, add_dim=args.add_dim, add_dino=args.add_dino)

    vae_state_dict = vae.state_dict()
    vae_ori_state_dict = vae_ori.state_dict()
    for name, param in vae_state_dict.items():
        if name in vae_ori_state_dict:
            if vae_ori_state_dict[name].shape == param.shape:
                vae_state_dict[name].copy_(vae_ori_state_dict[name])
            else:
                print(f"skip {name} as shape not match")
        else:
            print(f"skip {name} as not exist in vae_ori")
    vae.load_state_dict(vae_state_dict)

    def _replace_vae_conv():
        # replace the first layer to accept 27 in_channels
        _weight = vae.encoder.conv_in.weight.clone()
        _bias = vae.encoder.conv_in.bias.clone()
        _weight = _weight.repeat((1, 9, 1, 1))  # Keep selected channel(s)
        # reduce the activation magnitude
        _weight *= 0.1
        # new conv_in channel
        _n_convin_out_channel = vae.encoder.conv_in.out_channels

        _new_conv_in = Conv2d(
            27, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        vae.encoder.conv_in = _new_conv_in
        vae.config.in_channels = 27
        vae.config["in_channels"] = 27
        logging.info("vae conv_in layer is replaced")

        # replace the out layer to out 27 in_channels
        _weight = vae.decoder.conv_out.weight.clone()  # [4, 320, 3, 3]
        _bias = vae.decoder.conv_out.bias.clone()  # [4]
        _weight = _weight.repeat((9, 1, 1, 1))  # Keep selected channel(s)
        # reduce the activation magnitude
        _weight *= 0.1
        # copy bias
        _bias = _bias.repeat(9)
        # new conv_out channel
        _n_convout_in_channel = vae.decoder.conv_out.in_channels

        _new_conv_out = Conv2d(
            _n_convout_in_channel, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        # replace config
        _new_conv_out.weight = Parameter(_weight)
        _new_conv_out.bias = Parameter(_bias)
        vae.decoder.conv_out = _new_conv_out
        vae.config.out_channels = 27
        vae.config["out_channels"] = 27
        logging.info("vae conv_out layer is replaced")

        return
    
    if args.super_reshape:
        _replace_vae_conv()

    # 1. fix VAE original parameters
    for param in vae.parameters():
        param.requires_grad = False

    # 2. unfix fuse_layer.parameters
    if args.add_dim:
        for param in vae.decoder.up_blocks[1].fuse_layer.parameters():
            param.requires_grad = True
        for param in vae.decoder.up_blocks[2].fuse_layer.parameters():
            param.requires_grad = True
        for param in vae.decoder.up_blocks[0].fuse_layer.parameters():
            param.requires_grad = True
        for param in vae.decoder.up_blocks[3].fuse_layer.parameters():
            param.requires_grad = True

    if args.super_reshape:
        for param in vae.encoder.conv_in.parameters():
            param.requires_grad = True
        for param in vae.decoder.conv_out.parameters():
            param.requires_grad = True

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_parameters = vae.parameters()
    if args.add_dim:
        trainable_parameters = (list(vae.decoder.up_blocks[0].fuse_layer.parameters()) + 
                                list(vae.decoder.up_blocks[1].fuse_layer.parameters()) + 
                                list(vae.decoder.up_blocks[2].fuse_layer.parameters()) +
                                list(vae.decoder.up_blocks[3].fuse_layer.parameters())
        )
        if args.super_reshape:
            trainable_parameters = trainable_parameters + list(vae.encoder.conv_in.parameters()) + list(vae.decoder.conv_out.parameters())

    optimizer = optimizer_cls(
                trainable_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
    
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    train_dataset = make_train_dataset(args, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataset = make_val_dataset(args, accelerator)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.add_dino:
        DINO_Net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
        DINO_Net.eval()
        DINO_Net.requires_grad = False
        DINO_Net.to(accelerator.device, dtype=weight_dtype)

        for param in DINO_Net.parameters():
            param.requires_grad = False

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_images")
        tracker_config.pop("validation_latent")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    logger.info(args)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        epoch_loss = 0.0
        l2_train_loss = 0.0
        l2_epoch_loss = 0.0
        p_train_loss = 0.0
        p_epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):

                conditioning_pixel_values = batch["conditioning_pixel_values"]
                if args.super_reshape:
                    conditioning_pixel_values = reshape_super_resolution(batch["conditioning_pixel_values"])

                condition_encode_features = None
                dino = None
                if args.add_dim:
                    condition_features = vae.module.encode(conditioning_pixel_values.to(weight_dtype))
                    condition_encode_features = condition_features.enc_feature_list
                    latent = (batch["latent_values"].to(weight_dtype)) / vae.module.config.scaling_factor
                    if args.add_dino:
                        DINO_patch_size = 14
                        H, W = batch["conditioning_pixel_values"][0].shape[-2:]
                        scale_factor = DINO_patch_size / 16
                        UpSample = nn.UpsamplingBilinear2d(
                            size=((int)(H * scale_factor), 
                                (int)(W * scale_factor)))
                        with torch.no_grad():
                            input_dino = UpSample(batch["conditioning_pixel_values"])
                        dino = DINO_Net.get_intermediate_layers(input_dino, 4, True)[3]

                    decode_result = vae.module.decode(latent, enc_features=condition_encode_features, dino=dino, return_dict=False)[0]
                else:
                    latent = (batch["latent_values"].to(weight_dtype)) / vae.config.scaling_factor
                    decode_result = vae.decode(latent, enc_features=condition_encode_features, return_dict=False)[0]

                if args.super_reshape:
                    decode_result = reverse_reshape_super_resolution(decode_result)

                target = batch["pixel_values"]

                loss, l2_loss, p_loss = dim_loss(target.float(), decode_result.float())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                epoch_loss += train_loss

                l2_avg_loss = accelerator.gather(l2_loss.repeat(args.train_batch_size)).mean()
                l2_train_loss += l2_avg_loss.item() / args.gradient_accumulation_steps
                l2_epoch_loss += l2_train_loss

                p_avg_loss = accelerator.gather(p_loss.repeat(args.train_batch_size)).mean()
                p_train_loss += p_avg_loss.item() / args.gradient_accumulation_steps
                p_epoch_loss += p_train_loss

                # Backpropagate
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"l2_train_loss": l2_train_loss}, step=global_step)
                accelerator.log({"p_train_loss": p_train_loss}, step=global_step)
                train_loss = 0.0
                l2_train_loss = 0.0
                p_train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        logger.info(f"epoch_loss:{epoch_loss}, step={global_step}")
        logger.info(f"l2_epoch_loss:{l2_epoch_loss}, step={global_step}")
        logger.info(f"p_epoch_loss:{p_epoch_loss}, step={global_step}")
        accelerator.log({"epoch_loss": epoch_loss}, step=epoch+1)

        if accelerator.is_main_process:
            if args.validation_prompts is not None and (epoch + 1) % args.validation_epochs == 0:
                if args.add_dino:
                    log_validation(
                        val_dataloader,
                        vae,
                        args,
                        accelerator,
                        weight_dtype,
                        epoch+1,
                        DINO_Net=DINO_Net
                    )
                else:
                    log_validation(
                        val_dataloader,
                        vae,
                        args,
                        accelerator,
                        weight_dtype,
                        epoch+1,
                    )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
