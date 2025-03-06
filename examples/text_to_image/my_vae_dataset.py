import os
import datasets
import pandas as pd
import numpy as np

class MyDataset(datasets.GeneratorBasedBuilder):
    # Dataset information
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                'conditioning_image': datasets.Image(),
                'latent': datasets.Array3D(shape=(4, 60, 60), dtype='float32'),
            }),
        )

    # Dataset split
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'text_filepath': 'ISTD+_Dataset/train_aug/train_vae.json',
                    'image_dir': 'ISTD+_Dataset/train_aug/shadow_free/',
                    'condition_image_dir': 'ISTD+_Dataset/train_aug/origin/',
                    'latent_dir': 'ISTD+_Dataset/train_aug/latents_sample',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'text_filepath': 'ISTD+_Dataset/test_aug/test_vae.json',
                    'image_dir': 'ISTD+_Dataset/test_aug/shadow_free/',
                    'condition_image_dir': 'ISTD+_Dataset/test_aug/origin/',
                    'latent_dir': 'ISTD+_Dataset/test_aug/latents_sample',
                },
            ),
        ]

    # generate_examples
    def _generate_examples(self, text_filepath, image_dir, condition_image_dir, latent_dir):
        metadata = pd.read_json(text_filepath, lines=True)

        for _, row in metadata.iterrows():

            image_path = row["image"]
            image_path = os.path.join(image_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(
                condition_image_dir, conditioning_image_path
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            latent_path = row["latent"]
            latent_path = os.path.join(latent_dir, latent_path)
            latent = np.load(latent_path).astype('float32')

            yield row["image"], {
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
                "latent": latent,
            }