from glob import glob
import os 
import pickle
import imageio
import torch 
import numpy as np
import time
import h5py

# original dataset
dataset_path = "/database/hanni/AIMET/DIV2K/DIV2K_train_HR"

# converted dataset
output_path = "/database/hanni/DIV2KS/DIV2K_train_HR"
os.makedirs(output_path, exist_ok=False)
png_files = sorted(glob(f"{dataset_path}/*.png"))

# PT Conversion

for image_path in png_files:
    if os.path.isfile(image_path):
        image = imageio.imread(image_path)
        pt_path = f"{output_path}/{os.path.basename(image_path).replace('.png', '.pt')}"
        with open(pt_path, 'wb') as f:
            pickle.dump(image, f)

# # HDF5 Conversion
# for image_path in png_files:
#     if os.path.isfile(image_path):
#         image = imageio.imread(image_path)
#         h5_path = f"{output_path}/{os.path.basename(image_path).replace('.png', '.h5')}"
#         with h5py.File(h5_path, 'w') as f:
#             f.create_dataset("image", data=image)