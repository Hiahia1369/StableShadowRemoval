# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonoimage.github.io
# --------------------------------------------------------------------------


from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch

from .image_util import get_tv_resample_method, resize_max_res


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two image maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist


def ensemble_shadow(
    image: torch.Tensor,
    scale_invariant: bool = True,
    shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ensembles image maps represented by the `image` tensor with expected shape `(B, 1, H, W)`, where B is the
    number of ensemble members for a given prediction of size `(H x W)`. Even though the function is designed for
    image maps, it can also be used with disparity maps as long as the input tensor values are non-negative. The
    alignment happens when the predictions have one or more degrees of freedom, that is when they are either
    affine-invariant (`scale_invariant=True` and `shift_invariant=True`), or just scale-invariant (only
    `scale_invariant=True`). For absolute predictions (`scale_invariant=False` and `shift_invariant=False`)
    alignment is skipped and only ensembling is performed.

    Args:
        image (`torch.Tensor`):
            Input ensemble image maps.
        scale_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as scale-invariant.
        shift_invariant (`bool`, *optional*, defaults to `True`):
            Whether to treat predictions as shift-invariant.
        output_uncertainty (`bool`, *optional*, defaults to `False`):
            Whether to output uncertainty map.
        reduction (`str`, *optional*, defaults to `"median"`):
            Reduction method used to ensemble aligned predictions. The accepted values are: `"mean"` and
            `"median"`.
        regularizer_strength (`float`, *optional*, defaults to `0.02`):
            Strength of the regularizer that pulls the aligned predictions to the unit range from 0 to 1.
        max_iter (`int`, *optional*, defaults to `2`):
            Maximum number of the alignment solver steps. Refer to `scipy.optimize.minimize` function, `options`
            argument.
        tol (`float`, *optional*, defaults to `1e-3`):
            Alignment solver tolerance. The solver stops when the tolerance is reached.
        max_res (`int`, *optional*, defaults to `1024`):
            Resolution at which the alignment is performed; `None` matches the `processing_resolution`.
    Returns:
        A tensor of aligned and ensembled image maps and optionally a tensor of uncertainties of the same shape:
        `(1, 1, H, W)`.
    """
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(f"Expecting 4D tensor of shape [B,1,H,W]; got {image.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    if not scale_invariant and shift_invariant:
        raise ValueError("Pure shift-invariant ensembling is not supported.")

    def init_param(image: torch.Tensor):
        init_min = image.reshape(ensemble_size, -1).min(dim=1).values
        init_max = image.reshape(ensemble_size, -1).max(dim=1).values

        if scale_invariant and shift_invariant:
            init_s = 1.0 / (init_max - init_min).clamp(min=1e-6)
            init_t = -init_s * init_min
            param = torch.cat((init_s, init_t)).cpu().numpy()
        elif scale_invariant:
            init_s = 1.0 / init_max.clamp(min=1e-6)
            param = init_s.cpu().numpy()
        else:
            raise ValueError("Unrecognized alignment.")

        return param

    def align(image: torch.Tensor, param: np.ndarray) -> torch.Tensor:
        if scale_invariant and shift_invariant:
            s, t = np.split(param, 2)
            s = torch.from_numpy(s).to(image).view(ensemble_size, 1, 1, 1)
            t = torch.from_numpy(t).to(image).view(ensemble_size, 1, 1, 1)
            out = image * s + t
        elif scale_invariant:
            s = torch.from_numpy(param).to(image).view(ensemble_size, 1, 1, 1)
            out = image * s
        else:
            raise ValueError("Unrecognized alignment.")
        return out

    def ensemble(
        image_aligned: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        uncertainty = None
        if reduction == "mean":
            prediction = torch.mean(image_aligned, dim=0, keepdim=True)
            if return_uncertainty:
                uncertainty = torch.std(image_aligned, dim=0, keepdim=True)
        elif reduction == "median":
            prediction = torch.median(image_aligned, dim=0, keepdim=True).values
            if return_uncertainty:
                uncertainty = torch.median(
                    torch.abs(image_aligned - prediction), dim=0, keepdim=True
                ).values
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")
        return prediction, uncertainty

    def cost_fn(param: np.ndarray, image: torch.Tensor) -> float:
        cost = 0.0
        image_aligned = align(image, param)

        for i, j in torch.combinations(torch.arange(ensemble_size)):
            diff = image_aligned[i] - image_aligned[j]
            cost += (diff**2).mean().sqrt().item()

        if regularizer_strength > 0:
            prediction, _ = ensemble(image_aligned, return_uncertainty=False)
            err_near = (0.0 - prediction.min()).abs().item()
            err_far = (1.0 - prediction.max()).abs().item()
            cost += (err_near + err_far) * regularizer_strength

        return cost

    def compute_param(image: torch.Tensor):
        import scipy

        image_to_align = image.to(torch.float32)
        if max_res is not None and max(image_to_align.shape[2:]) > max_res:
            image_to_align = resize_max_res(
                image_to_align, max_res, get_tv_resample_method("nearest-exact")
            )

        param = init_param(image_to_align)

        res = scipy.optimize.minimize(
            partial(cost_fn, image=image_to_align),
            param,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )

        return res.x

    requires_aligning = scale_invariant or shift_invariant
    ensemble_size = image.shape[0]

    if requires_aligning:
        param = compute_param(image)
        image = align(image, param)

    image, uncertainty = ensemble(image, return_uncertainty=output_uncertainty)

    image_max = image.max()
    if scale_invariant and shift_invariant:
        image_min = image.min()
    elif scale_invariant:
        image_min = 0
    else:
        raise ValueError("Unrecognized alignment.")
    image_range = (image_max - image_min).clamp(min=1e-6)
    image = (image - image_min) / image_range
    if output_uncertainty:
        uncertainty /= image_range

    return image, uncertainty  # [1,1,H,W], [1,1,H,W]
