import numpy as np
from numba import jit, float64, int64, void
import matplotlib.pyplot as plt

plt.ion()

GROUPNAME = 'Noise'

WHITE = 'White'
SALTPEPPER = 'Salt and Pepper'

OPS = (WHITE, SALTPEPPER)


def apply(in_image, operation, p1, p2, p3, p4):
    if operation == WHITE:
        try:
            sigma = float(p1)
        except ValueError:
            print("Value Error")
            return
        return white_noise(in_image, sigma)

    elif operation == SALTPEPPER:
        try:
            p = float(p1)
        except ValueError:
            print("Value Error")
            return
        return saltpepper_noise(in_image, p)


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def white_noise(image, sigma):
    h, w, d = image.shape
    noise = np.random.randn(h, w, d) * sigma
    image_with_noise = image + noise
    for x in range(h):
        for y in range(w):
            for z in range(d):
                if image_with_noise[x, y, z] < 0.0:
                    image_with_noise[x, y, z] = 0.0
                if image_with_noise[x, y, z] > 255.0:
                    image_with_noise[x, y, z] = 255.0
    return image_with_noise


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def saltpepper_noise(image, probability):
    h, w, d = image.shape
    for x in range(h):
        for y in range(w):
            for z in range(d):
                if np.random.random() < probability:
                    if np.random.random() < 0.5:
                        image[x, y, z] = 255.0
                    else:
                        image[x, y, z] = 0.0
    return image

@jit(float64[:, :, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def add_mixed_noise(img, sigma_white, probability_salt_pepper):
    img = img.copy()
    h, w, d = img.shape

    for x in range(h):
        for y in range(w):
            for z in range(d):
                if y < w / 2:
                    v = img[x, y, z] + np.random.randn(1)[0] * sigma_white
                    v = max(min(255.0, v), 0.0)
                    img[x, y, z] = v
                elif y == w:
                    img[x, y, z] = 255.0
                else:
                    if np.random.random() < probability_salt_pepper:
                        if np.random.random() < (probability_salt_pepper / 2):
                            img[x, y, z] = 255.0
                        else:
                            img[x, y, z] = 0.0
    return img