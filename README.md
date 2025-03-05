# StableShadowRemoval
This is the official implementation of the paper [Detail-Preserving Latent Diffusion for Stable Shadow Removal](https://arxiv.org/abs/2412.17630).

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-istd)](https://paperswithcode.com/sota/shadow-removal-on-istd?p=shadowformer-global-context-helps-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-adjusted-istd)](https://paperswithcode.com/sota/shadow-removal-on-adjusted-istd?p=shadowformer-global-context-helps-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-srd)](https://paperswithcode.com/sota/shadow-removal-on-srd?p=shadowformer-global-context-helps-image) -->

<!-- #### News
* **Feb 24, 2024**: Release the pretrained models for ISTD and ISTD+.
* **Feb 18, 2024**: Release the training and testing codes.
* **Feb 17, 2024**: Add the testing results and the description of our work. -->

## Introduction
We propose a two-stage fine-tuning pipeline to transform a pre-trained Stable Diffusion model into an image-conditional shadow-free image generator. This approach enables robust, high-resolution shadow removal without an input shadow mask.We introduce a shadow-aware detail injection module that utilizes the VAE encoder features to modulate the pre-trained VAE decoders, selectively aligning per-pixel details from the input image with those in the output shadow-free image.

For more details, please refer to our [original paper](https://arxiv.org/abs/2410.01719).


<p align=center><img width="80%" src="doc/pipeline.png "/></p>


## Requirement
* Python 3.10
* CUDA 11.7
```bash
cd diffusers
pip install -e .
cd examples/text_to_image/
pip install -r requirements.txt
```
And initialize an Accelerate environment with:
```bash
accelerate config
```
Or for a default accelerate configuration without answering questions about your environment:
```bash
accelerate config default
```
## Datasets
* ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)
* SRD [[link]](https://github.com/vinthony/ghost-free-shadow-removal)
* INS [[link]](https://1drv.ms/f/c/293105fdd25c43e1/Ehs2NWKPVmFPrnuudfVUM8EBC3DzwOuKTcm_kmmM4h17dg?e=Dqpb7u)
* Real Photo [[link]](https://1drv.ms/u/c/293105fdd25c43e1/Ea3oVBYVNGFPm5R2DSh5-54BvhSpkChYoYyuDjk54vERgw?e=caZWik)
* WSRD+ [[link]](https://github.com/movingforward100/Shadow_R)
## Pretrained models
[ISTD+](https://1drv.ms/f/c/293105fdd25c43e1/Ehi3WVIAfEJOu1Q4lwbVSaQB3Ukz441XRIJrIKZsl7rwAg?e=XcM0Pa) | [SRD](https://1drv.ms/f/c/293105fdd25c43e1/Ev3_e4zkLxxGmuEu2aJ9WWgBTvDh7JipOxogGWz1Do946A?e=pAuQRJ) | [INS](https://1drv.ms/f/c/293105fdd25c43e1/EjxvJtp-MGpAnnCPj_-6nJ4BLZnVgqAgRwnHAwIHNYDaIg?e=EwL6Vq)  |  [WSRD+](https://1drv.ms/f/c/293105fdd25c43e1/EgqB3_8HuINJlaC2AvUsABABkm2L0YvBMfpD9yI4X2Cmzg?e=iHaOLr)

Please download the corresponding pretrained model and modify the `unet_path` and `vae_path` in `examples/text_to_image/inference.py`.

## Test
You can directly test the performance of the pre-trained model as follows
1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `examples/text_to_image/inference.py` :
```python
unet_path  # pretrained stage-one unet weight path -- Line 18
vae_path  # pretrained stage-two dim weight path --Line 19
image_folder  # input data path -- Line 21
result_dir   #result output path --Line 23
```
2. Test the model
```python
python inference.py
```

## Train
### Stage one
1. Download datasets and set the following structure, and modify the dataset path in the `examples/text_to_image/my_dataset.py`.
```
|-- ISTD+_Dataset
    |-- train
        |-- origin  # shadow image
        |-- shadow_free  # shadow-free image GT
        |-- train.json  # text_file
    |-- test
        |-- origin  # shadow image
        |-- shadow_free  # shadow-free image GT
        |-- test.json  # text_file
```
```python
text_filepath  # text_file path
image_dir  # shadow-free image GT path
condition_image_dir  # shadow image path
```
* `text_file` can be generated by `examples/text_to_image/json_generate.py`, set `is_stage_1=True`.

2. The training file is `examples/text_to_image/train_text_to_image.py`, Use the following command to train and set optional parameters:
```bash
./train.sh
```
```python
CUDA_VISIBLE_DEVICES="0,1"  # Select GPU
num_processes=2  # Set the number of GPUs
mixed_precision="fp16"
pretrained_model_name_or_path  # pretrained stable diffusion path
train_data_dir  # dataset split file path
prediction_type  # Set to sample, diffusion predict the image latent instead of noise
```

### Stage two
1. Use the model trained in the stage-one to generate the latent of the shadow-free image and set optional parameters:
```python
python inference.py 
```
```python
unet_path='trained in the stage-one'
vae_path=stabilityai/stable-diffusion-2
output_type=latent  #Line 42
```
2. Set the dataset to the following structure, and modify the dataset path in the `examples/text_to_image/my_vae_dataset.py`:
```
|-- ISTD+_Dataset
    |-- train
        |-- origin  # shadow image
        |-- shadow_free  # shadow-free image GT
        |-- latents_sample  # predicted shadow-free latent
        |-- train_vae.json  # text_file
    |-- test
        |-- origin  # shadow image
        |-- shadow_free  # shadow-free GT
        |-- latents_sample  # predicted shadow-free latent
        |-- test_vae.json  # text_file
```
```python
text_filepath  # text_file path
image_dir  # shadow-free image GT path
condition_image_dir  # shadow image path
latent_dir  # predicted shadow-free latent path
```
* `text_file` can be generated by `examples/text_to_image/json_generate.py`, set `is_stage_1=False`.

3. The training file is `examples/text_to_image/train_vae_decoder.py`, Use the following command to train and set optional parameters:
```bash
./train_vae.sh
```
```python
add_cfw=true  #add detail injection model
add_dino=true  #add dino feature
```
Since the image latent for the stage-two training needs to be generated in advance, it is necessary to first perform data augmentation on the image and then generate the latent.

## Large-size inputs
### Train
#### Stage one
Downscale the input images to `W/k × H/k` for training, with `k = 3` for the WSRD+ dataset.
#### Stage two
Use stage-one model to generate the latent of the downscaled image, while the VAE encoder input the original-size image.
set `train_vae.sh` optional parameters:
```python
super_reshape=true
super_reshape_k=3  #set reshape k
```
### Test
1. Use the stage-one model to generate the latent of the downscaled image:
```python
python inference.py
```
2. Generate the final result by combining the latent of the downscaled image with the original-size image:
```python
python inference_vae.py
```

## Evaluation
The results reported in the paper are calculated by the `matlab` script used in [previous method](https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal/tree/master/codes). Details refer to `evaluation/measure_shadow.m`.


## Results
#### Evaluation on ISTD+, SRD and INS

| Datasets | PSNR | SSIM |
| :-- | :--: | :--: | 
| ISTD+ |  35.19 | 0.974 | 
| SRD | 33.63 | 0.968 |
| INS | 30.56 | 0.975 | 
| WSRD+ | 26.26 | 0.827 | 

<!-- #### Visual Results
<p align=center><img width="80%" src="doc/res.jpg"/></p> -->

#### Testing results
The testing results on dataset ISTD+, SRD,  INS and WSRD+ are: [results](https://1drv.ms/f/c/293105fdd25c43e1/Er5Fb6_lU3NMsk4_7tjA90IBCZs7_iKBdY9qWBxZ4kEMQw?e=XhgkFF).

## References
Our implementation is based on [Diffusers](https://github.com/huggingface/diffusers). We would like to thank them.

## Citation
Bibtex:
```
@inproceedings{xu2025stableshadowremoval,
title={Detail-Preserving Latent Diffusion for Stable Shadow Removal},
author={Xu, Jiamin and Zheng, Yuxin and Li, Zelong and Wang, Chi and Gu, Renshu and Xu, Weiwei and Xu, Gang},
booktitle={Proceedings of the CVPR Conference on Artificial Intelligence},
year={2025}
```

## Contact
If you have any questions, please contact 2451773098@qq.com.
