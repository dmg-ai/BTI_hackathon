import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ssl

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics
import torch
import wandb
from dataset import Dataset
from torch.utils.data import DataLoader
from utils import *

ssl._create_default_https_context = ssl._create_unverified_context

ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["walls", "windows", "doors"]
ACTIVATION = None  #'sigmoid'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SHAPE = 512
BATCH_SIZE = 16
LOSS_NAME = "DICE"

x_train_dir = "my_train/new_train/image/"
y_train_dir = "my_train/new_train/mask/"

x_valid_dir = "my_train/new_valid/image/"
y_valid_dir = "my_train/new_valid/mask/"

exp_folder = f"experiments/new_data/checkpoints_{RESIZE_SHAPE}x{RESIZE_SHAPE}_{ENCODER}_{ACTIVATION}_{LOSS_NAME}_unet"
os.makedirs(exp_folder)

wandb.init(
    project="crym",
    name=f"{ENCODER}_{RESIZE_SHAPE}x{RESIZE_SHAPE}_{LOSS_NAME}",
    config={
        "model": "Unet",
        "epochs": 50,
        "batch_size": BATCH_SIZE,
        "encoder": ENCODER,
        "experiment": "exp9",
    },
)


def get_training_augmentation():
    train_transform = [
        A.Resize(RESIZE_SHAPE, RESIZE_SHAPE),
        A.CLAHE(p=1),
        # A.HorizontalFlip(p=0.5),
        # A.Rotate(p=0.5)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():

    test_transform = [
        A.Resize(RESIZE_SHAPE, RESIZE_SHAPE),
        A.CLAHE(p=1),
        A.PadIfNeeded(384, 384),
    ]
    return A.Compose(test_transform)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

loss = smp.losses.DiceLoss(mode="multilabel")
loss.__name__ = LOSS_NAME
# loss = smp.losses.SoftBCEWithLogitsLoss()
# loss.__name__ = LOSS_NAME

metrics = [smp.utils.metrics.IoU()]

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0
for i in range(0, 50):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if max_score < valid_logs["iou_score"]:
        max_score = valid_logs["iou_score"]
        torch.save(
            model, os.path.join(exp_folder, f"best_model_{max_score}_{i}-epoch.pth")
        )
        print("Model saved!")

    wandb.log(
        {
            f"train/{loss.__name__}": np.float64(train_logs[loss.__name__]),
            f"val/{loss.__name__}": np.float64(valid_logs[loss.__name__]),
            f"train/IoU": np.float64(train_logs[metrics[0].__name__]),
            f"val/IoU": np.float64(valid_logs[metrics[0].__name__]),
        }
    )

    if i == 12:
        optimizer.param_groups[0]["lr"] = 1e-5
        print("Decrease decoder learning rate to 1e-5!")

wandb.finish()
