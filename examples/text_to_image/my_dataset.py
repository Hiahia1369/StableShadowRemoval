import os
import datasets
import pandas as pd
import numpy as np

class MyDataset(datasets.GeneratorBasedBuilder):
    # Dataset information
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'text': datasets.Value('string'),
                'image': datasets.Image(),
                'conditioning_image': datasets.Image(),
            }),
        )

    # Dataset split
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'text_filepath': 'ISTD+_Dataset/train/train.json',
                    'image_dir': 'ISTD+_Dataset/train/shadow_free/',
                    'condition_image_dir': 'ISTD+_Dataset/train/origin/',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'text_filepath': 'ISTD+_Dataset/test/test.json',
                    'image_dir': 'ISTD+_Dataset/test/shadow_free/',
                    'condition_image_dir': 'ISTD+_Dataset/test/origin/',
                },
            ),
        ]

    # generate_examples
    def _generate_examples(self, text_filepath, image_dir, condition_image_dir):
        metadata = pd.read_json(text_filepath, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image_path = os.path.join(image_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(
                condition_image_dir, conditioning_image_path
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }