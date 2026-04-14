import cv2
import numpy as np
import random


def random_flip(img, steering):
    # flip image horizontally and reverse steering angle
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def random_brightness(img, steering):
    # randomly darken or brighten the image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.4, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, steering


def random_zoom(img, steering):
    # zoom in randomly to simulate distance variation
    h, w = img.shape[:2]
    zoom = random.uniform(1.0, 1.3)
    zh = int(h / zoom)
    zw = int(w / zoom)
    top = random.randint(0, h - zh)
    left = random.randint(0, w - zw)
    img = img[top:top + zh, left:left + zw]
    img = cv2.resize(img, (w, h))
    return img, steering


def random_pan(img, steering):
    # shift image left/right/up/down slightly
    h, w = img.shape[:2]
    tx = w * random.uniform(-0.1, 0.1)
    ty = h * random.uniform(-0.1, 0.1)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h))
    steering += tx / w * 0.3
    return img, steering


def augment_image(img, steering):
    # apply random augmentations to help the model generalize
    img, steering = random_flip(img, steering)

    funcs = random.sample([random_brightness, random_zoom, random_pan],
                          k=random.randint(0, 2))
    for fn in funcs:
        img, steering = fn(img, steering)

    return img, steering
