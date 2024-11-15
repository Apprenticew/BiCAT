# copyright: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

import random
import math
import torch
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch.nn.functional as F


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    # v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    # color = (125, 123, 114)
    # color = (0, 0, 0)
    color = (0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def random_crop_and_resize(img, v, ):
    # 打开图像

    img_width, img_height = img.size
    v1 = int((1 - v) * img.size[0])
    v2 = int((1 - v) * img.size[1])
    # 计算随机裁剪的位置
    left = random.randint(0, img_width - v1)
    top = random.randint(0, img_height - v2)
    right = left + v1
    bottom = top + v2

    # 裁剪图像
    cropped_img = img.crop((left, top, right, bottom))

    # 将裁剪后的图像拉伸到原图大小
    resized_img = cropped_img.resize((img_width, img_height), PIL.Image.LANCZOS)
    return resized_img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -8, 8),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


def augment_list_no_cut():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -8, 8),
        (Sharpness, 0.05, 0.95),
        (Solarize, 0, 256),
    ]
    return l


def augment_list_wo_color(v):
    # v = 0.2
    l = [
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -4, 4),
        (Sharpness, 0.05, 0.95),
        # (ShearX, -0.1, 0.1),
        # (ShearY, -0.1, 0.1),
        # (TranslateX, -0.1, 0.1),
        # (TranslateY, -0.1, 0.1),
        # (random_crop_and_resize, 0, v),
        (ShearX, -v, v),
        (ShearY, -v, v),
        (TranslateX, -v, v),
        (TranslateY, -v, v),
    ]
    return l


def add_speckle_noise(image, var, p):
    """
    添加 speckle 噪声到图像中。

    Args:
        image (numpy.ndarray): 输入图像。
        var (float): 噪声方差。

    Returns:
        numpy.ndarray: 添加噪声后的图像。
    """

    image = np.array(image) / 255.0
    height, width = image.shape
    noise = np.random.randn(height, width) * np.sqrt(var)
    # noise = np.random.normal(0, var ** 0.5, image.shape)
    noisy_image = image + np.power(image, p) * noise
    # 确保输出值在 [0,1] 范围内
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    # 将NumPy数组转换回PIL图像，并缩放回原始像素范围
    noisy_image = (noisy_image * 255).astype(np.uint8)
    noisy_image = PIL.Image.fromarray(noisy_image)

    return noisy_image


class RandAugment:
    def __init__(self, n, v=0.1):
        self.n = n
        self.v = v

        self.augment_list = augment_list_wo_color(v)

    def __call__(self, img):
        if random.random() < 0.2:
            var = 0 + float(0.05) * random.random()
            p = 0.45 + float(0.1) * random.random()
            img = add_speckle_noise(img, var, p)

        cutout_val = random.random() * 0.2
        img = Cutout(img, cutout_val)

        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)

        return img
