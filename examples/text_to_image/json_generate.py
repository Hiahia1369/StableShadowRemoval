import os
import json

# image folder
image_folder = "image_folder"
image_dir = os.path.join(image_folder, "shadow_free")
condition_image_dir = os.path.join(image_folder, "origin")
latents_dir = os.path.join(image_folder, "latents_sample")

# output path
output_file = os.path.join("output path", "train.json")

# get image
image_files = sorted(os.listdir(image_dir))
condition_image_files = sorted(os.listdir(condition_image_dir))
latent_files = sorted(os.listdir(latents_dir))

# assert image num
assert len(image_files) == len(condition_image_files), "The number of files in the two folders is inconsistent"

entries = []
is_stage_1 = True  #stage_2 false
if is_stage_1:
    for image_file, condition_image_file in zip(image_files, condition_image_files):
        entry = {
            "text": "",  # Fill with empty text
            "image": image_file,
            "conditioning_image": condition_image_file
        }
        entries.append(entry)
else:
    for image_file, condition_image_file, latent_file in zip(image_files, condition_image_files, latent_files):
        entry = {
            "image": image_file,
            "conditioning_image": condition_image_file,
            "latent": latent_file
        }
        entries.append(entry)

# write json
with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f)
            f.write('\n')

print(f"JSON file has been generated: {output_file}")
