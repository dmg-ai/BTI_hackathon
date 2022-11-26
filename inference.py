import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import skimage
import torch
from matplotlib import pyplot as plt
from preprocessing import *
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

ENCODER = "se_resnext50_32x4d"  #'efficientnet-b2'
ENCODER_WEIGHTS = "imagenet"
MODEL_PATH = "experiments/new_data/checkpoints_512x512_se_resnext50_32x4d_None_BCE_unet/best_model_0.5967634263876324_31-epoch.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SHAPE = 512
ACTIVATION = None


test_names = [os.path.join("test/", file) for file in os.listdir("test/")]

model = torch.load(MODEL_PATH)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

save_path = os.path.join("/".join(MODEL_PATH.split("/")[:-1]), "result_mask")
print(save_path)
for filepath in tqdm(test_names):
    # print(filepath)
    x_tensor, img_shape = get_image(filepath, (512, 512), preprocessing_fn, DEVICE)
    mask = get_model_predicts(model, x_tensor, ACTIVATION)

    resized_mask = skimage.transform.resize(
        mask,
        img_shape,
        mode="edge",
        anti_aliasing=False,
        anti_aliasing_sigma=None,
        preserve_range=True,
        order=0,
    )

    res_mask = np.zeros(resized_mask.shape[:2])
    for ax in range(3):
        res_mask[resized_mask[:, :, ax] > 0] = ax + 1
    res_mask = res_mask.astype(np.uint8)
    os.makedirs(save_path, exist_ok=True)
    skimage.io.imsave(os.path.join(save_path, filepath.split("/")[1]), res_mask)
