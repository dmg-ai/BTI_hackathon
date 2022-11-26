import albumentations as A
import cv2
import numpy as np
import torch


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def get_image(image_path: str, image_size: tuple, preprocessing_fn, DEVICE="cpu"):
    image = cv2.imread(image_path)
    img_shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = preprocessing_fn(image)
    image = to_tensor(image)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    return x_tensor, img_shape


def get_model_predicts(model, x_tensor, ACTIVATION):

    predicted_mask = model.predict(x_tensor)
    if ACTIVATION is None:
        predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = predicted_mask.squeeze().cpu().numpy()  # .round())
    predicted_mask = np.where(predicted_mask > 0.5, 1, 0)
    predicted_mask = predicted_mask.transpose(1, 2, 0)

    return predicted_mask
