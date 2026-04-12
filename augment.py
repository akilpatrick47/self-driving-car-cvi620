import cv2
import numpy as np
import random


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentation functions
# Each receives a BGR image + steering angle, returns augmented versions.
# ─────────────────────────────────────────────────────────────────────────────

def random_flip(img: np.ndarray, steering: float):
    """
    Horizontally flip the image 50 % of the time.
    When flipped the steering angle must be negated.
    """
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def random_brightness(img: np.ndarray, steering: float):
    """
    Randomly adjust brightness by converting to HSV and
    scaling the V channel.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.4, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img, steering


def random_zoom(img: np.ndarray, steering: float):
    """
    Random zoom-in crop between 1× and 1.3× magnification.
    """
    h, w = img.shape[:2]
    zoom = random.uniform(1.0, 1.3)
    zh = int(h / zoom)
    zw = int(w / zoom)
    top  = random.randint(0, h - zh)
    left = random.randint(0, w - zw)
    img = img[top:top + zh, left:left + zw]
    img = cv2.resize(img, (w, h))
    return img, steering


def random_pan(img: np.ndarray, steering: float):
    """
    Randomly translate the image horizontally/vertically by up to 10 % of
    width/height. Adjust steering proportionally for horizontal shift.
    """
    h, w = img.shape[:2]
    tx = w * random.uniform(-0.1, 0.1)
    ty = h * random.uniform(-0.1, 0.1)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h))
    steering += tx / w * 0.3   # small proportional correction
    return img, steering


# ─────────────────────────────────────────────────────────────────────────────
# Batch augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Pool of augmentation functions (excluding flip, applied separately)
_AUG_POOL = [random_brightness, random_zoom, random_pan]


def augment_image(img: np.ndarray, steering: float):
    """
    Apply a random subset of augmentation techniques to one image.
    Flip is always considered; 1-2 additional transforms are applied at random.

    Parameters
    ----------
    img      : np.ndarray  BGR image (raw, before preprocessing)
    steering : float

    Returns
    -------
    img      : np.ndarray  augmented BGR image
    steering : float       (possibly modified)
    """
    # Always consider flip
    img, steering = random_flip(img, steering)

    # Randomly apply 0–2 additional augmentations
    funcs = random.sample(_AUG_POOL, k=random.randint(0, 2))
    for fn in funcs:
        img, steering = fn(img, steering)

    return img, steering
