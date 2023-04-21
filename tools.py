import numpy as np
from numba import jit, float64, int64, void, types, boolean
import matplotlib.pyplot as plt
import imageio

plt.ion()

GROUPNAME = 'Tools'

EXTRACT_CHANNEL = 'Extract channel'

INVERSION = 'Inversion'
HISTOGRAM_NORMALISATION = 'Hist. Normalisation'
HISTOGRAM_EQUALIZATION = 'Hist. Equalization'
LOCAL_HISTOGRAM_EQUALIZATION = 'Local Hist. Equalization'
EXTEND = 'Extend'
RESCALE = 'Rescale'
SAMPLE_UP = 'Sample Up'
BINARY = 'To Binary'
RGB2HSV = 'RGB To HSV'
RGB2HSL = 'RGB To HSL'
RGB2LAB = 'RGB To LAB'
RGB2PCA = 'RGB To PCA'
FOURIER = 'Fourier Transformation'
VALUE_NORMALISATION = 'Value Normalisation'

OPS = (EXTRACT_CHANNEL, HISTOGRAM_NORMALISATION, HISTOGRAM_EQUALIZATION, LOCAL_HISTOGRAM_EQUALIZATION,
       VALUE_NORMALISATION, INVERSION, EXTEND, RESCALE, RGB2HSV, RGB2HSL, RGB2LAB, BINARY, RGB2PCA, FOURIER)


def apply(image, operation, p1, p2, p3, p4):
    if operation == EXTRACT_CHANNEL:
        channel = int(p1)
        return get_channel(image, channel)

    if operation == HISTOGRAM_NORMALISATION:
        p = float(p1)
        return histogram_normalisation(image, p)

    elif operation == HISTOGRAM_EQUALIZATION:
        alpha = float(p1)
        if image.shape[2] == 3:
            hsv_image = rgb2hsv(image)
            val = hsv_image[:, :, 2:3:] * 255
            val_eq = histogram_equalization(val, alpha)
            hsv_image[:, :, 2:3:] = val_eq / 255
            image = hsv2rgb(hsv_image)
        else:
            image = histogram_equalization(image, alpha)
        return image

    elif operation == LOCAL_HISTOGRAM_EQUALIZATION:
        alpha = float(p1)
        M = int(p2)
        if image.shape[2] == 3:
            hsv_image = rgb2hsv(image)
            val = hsv_image[:, :, 2:3:] * 255
            val_eq = local_histogram_equalization(val, alpha, M)
            hsv_image[:, :, 2:3:] = val_eq / 255
            image = hsv2rgb(hsv_image)
        else:
            image = local_histogram_equalization(image, alpha, M)
        return image

    elif operation == VALUE_NORMALISATION:
        return value_normalisation(image)

    elif operation == INVERSION:
        result = invert(image)
        return result

    elif operation == EXTEND:
        N = int(float(p1))

        if p2 == 'same':
            return extend_same(image, N)
        else:
            return extend_with_zeros(image, N)

    elif operation == RESCALE:
        factor = float(p1)
        return rescale(image, factor)

    elif operation == RGB2HSV:
        hsv_image = rgb2hsv(image)
        if p1 == 'h':
            return hsv_image[:, :, 0:1:] * 255 / 360
        elif p1 == 's':
            return hsv_image[:, :, 1:2:] * 255
        elif p1 == 'v':
            return hsv_image[:, :, 2:3:] * 255
        elif p1 == 'min sv':
            return np.minimum(hsv_image[:, :, 2:3:] * 255,hsv_image[:, :, 1:2:] * 255)
        elif p1 == 'all':
            hsv_image[:, :, 0] /= 360
            return hsv_image * 255
        elif p1 == 'hue swap':
            hsv_image[:, :, 0] += 180
            hsv_image[:, :, 0] %= 360
            return hsv2rgb(hsv_image)
        elif p1 == 'full sv':
            hsv_image[:, :, 1:3] = 1
            return hsv2rgb(hsv_image)
        elif p1 == 'full s':
            hsv_image[:, :, 1] = 1
            return hsv2rgb(hsv_image)
        elif p1 == 'full v':
            hsv_image[:, :, 2] = 1
            return hsv2rgb(hsv_image)
        elif p1 == 'hsvmap':
            hsv_image = np.zeros((360, 360, 3), dtype=np.float64)
            for h in range(360):
                for y in range(360):
                    if y >= 180:
                        s = (360 - y) / 180
                        v = 1.0
                    else:
                        s = 1.0
                        v = (y) / 180
                    hsv_image[y, h, :] = (h, s, v)
            return hsv2rgb(hsv_image)
        else:
            raise ValueError

    elif operation == RGB2HSL:
        hsl_image = rgb2hsl(image)
        if p1 == 'h':
            return hsl_image[:, :, 0:1:] * 255 / 360
        elif p1 == 's':
            return hsl_image[:, :, 1:2:] * 255
        elif p1 == 'l':
            return hsl_image[:, :, 2:3:] * 255
        elif p1 == 'all':
            hsl_image[:, :, 0] /= 360
            return hsl_image * 255
        else:
            raise ValueError

    elif operation == RGB2LAB:
        lab_image = rgb2lab(image)
        if p1 == 'l':
            return normalize(lab_image[:, :, 0:1:])
        elif p1 == 'a':
            return normalize(lab_image[:, :, 1:2:])
        elif p1 == 'b':
            return normalize(lab_image[:, :, 2:3:])
        elif p1 == 'all':
            return normalize(lab_image)
        else:
            raise ValueError

    elif operation == BINARY:
        binary = convert_to_binary(image)
        return normalize(binary.reshape((image.shape[0], image.shape[1], 1)).astype(np.float64))

    elif operation == RGB2PCA:
        return rgb2pca(image)

    elif operation == FOURIER:
        return normalize(fourier_transformation(image))


# ===========IMAGE IO =========================

def imread3D(path):
    """Reads an image from disk. Returns the array representation.

        Parameters
        ----------
        path : str
            Path to file (including file extension)

        Returns
        -------
        img : ndarray of float64
            Image as 3D array

        Notes
        -----
        'img' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    img = imageio.imread(path)  # first use scipys imread()
    if img.ndim == 2:
        h, w = img.shape
        img = img.reshape((h, w, 1)).astype(np.float64)  # if image has two dimensions, we add one dimension
    else:
        if np.all(img[:,:,0] == img[:,:,1])and np.all(img[:,:,0] == img[:,:,2]):
            return img[:,:,0:1:].astype(np.float64)
        h, w, d = img.shape
        if d == 4:
            img = img[:, :, :3]  # if image has 3 dimensions and 4 channels, drop last channel
    return img.astype(np.float64)

def imsave3D(path, img):
    """Saves the array representation of an image to disk.

        Parameters
        ----------
        path : str
            Path to file (including file extension)
        img : ndarray of float64
            Array representation of an image

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """
    assert img.ndim == 3
    h, w, d = img.shape
    assert d in {1, 3}
    if d == 1:
        imageio.imsave(path, img.reshape(h, w))
    else:
        imageio.imsave(path, img)


# ================== OTHERS ========================

@jit(int64[:, :](int64, int64, int64, int64), nopython=True, cache=True)
def get_valid_neighbours(h, w, x, y):
    has_upper = x > 0
    has_lower = x < h - 1
    has_left = y > 0
    has_right = y < w - 1

    neighbours = np.zeros((8, 2), dtype=np.int64)
    neighbour_index = 0

    if has_upper:
        neighbours[neighbour_index, :] = (x - 1, y)
        neighbour_index += 1
    if has_lower:
        neighbours[neighbour_index, :] = (x + 1, y)
        neighbour_index += 1
    if has_left:
        neighbours[neighbour_index, :] = (x, y - 1)
        neighbour_index += 1
    if has_right:
        neighbours[neighbour_index, :] = (x, y + 1)
        neighbour_index += 1
    if has_right and has_upper:
        neighbours[neighbour_index, :] = (x - 1, y + 1)
        neighbour_index += 1
    if has_left and has_upper:
        neighbours[neighbour_index, :] = (x - 1, y - 1)
        neighbour_index += 1
    if has_left and has_lower:
        neighbours[neighbour_index, :] = (x + 1, y - 1)
        neighbour_index += 1
    if has_right and has_lower:
        neighbours[neighbour_index, :] = (x + 1, y + 1)
        neighbour_index += 1

    return neighbours[:neighbour_index, :]


@jit(float64[:](), nopython=True, cache=True)
def get_random_unit3():
    r = np.random.random(3).astype(np.float64)
    return r / np.linalg.norm(r)


@jit(float64[:](float64), nopython=True, cache=True)
def get_saturated_color(hue):
    c = 1.0
    x = 1 - abs((hue / 60) % 2 - 1)
    if hue < 60:
        r_, g_, b_ = c, x, 0
    elif hue < 120:
        r_, g_, b_ = x, c, 0
    elif hue < 180:
        r_, g_, b_ = 0, c, x
    elif hue < 240:
        r_, g_, b_ = 0, x, c
    elif hue < 300:
        r_, g_, b_ = x, 0, c
    else:
        r_, g_, b_ = c, 0, x

    return np.array([r_, g_, b_], dtype=np.float64) * 255


@jit(float64[:](), nopython=True, cache=True)
def get_random_color():
    return get_saturated_color(np.random.random() * 360)


@jit(float64[:, :](int64), nopython=True, cache=True)
def get_saturated_colors(num_colors):
    colors = np.zeros((num_colors, 3), dtype=np.float64)
    for i in range(num_colors):
        if i == 0:
            colors[i, :] = np.ones(3, dtype=np.float64) * 255
        else:
            hue_i = 57 * i
            colors[i, :] = get_saturated_color(hue_i % 360) * (np.sin(i) / 4 + 0.75)
    return colors


# ================ BASIC TOOLS ====================

@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def invert(image):
    return 255.0 - image


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def normalize(img):
    #
    #  |-5  3 12|  +5  |0  8 17| *255/13 | 0 120 255|
    #  |-1 -2  1|  ->  |4  3  6|   ->    |60  45  90|
    #  |-1  4  6|      |4  9 11|         |60 135 165|
    #

    min_value = np.min(img)
    max_value = np.max(img)

    assert max_value != 0, "Maximum value of image is zero"

    return (img - min_value) * 255 / (max_value-min_value) # creates a copy


@jit(float64[:, :, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def trim(matrix, min_value, max_value):
    h, w, d = matrix.shape
    result = np.copy(matrix)
    for x in range(h):
        for y in range(w):
            for z in range(d):
                v = matrix[x, y, z]
                if v < min_value:
                    result[x, y, z] = min_value
                elif v > max_value:
                    result[x, y, z] = max_value
    return result


# ================= TRANSFORMATION ================

@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def fourier_transformation(image):
    # radius = Image.shape[0]/2
    # for x in range(0, Image.shape[0]):
    #  for y in range(0, Image.shape[1]):
    #          d = min(((x-radius)**2+(y-radius)**2)**0.5 /radius,1.0)
    #          Image[x,y] =d *0.5 + (1-d)*Image[x,y]

    M, N, d = image.shape
    quotient = 1.0 / np.sqrt(M * N)
    imagFac = -2 * np.pi * 1j

    f_size = int(M / 2)
    f_shape = (2 * f_size + 1, 2 * f_size + 1)

    F = np.zeros(f_shape, dtype=np.complex128)

    for m in range(-f_size, f_size):
        for n in range(0, f_size):
            sum = 0.0j
            for x in range(0, M):
                for y in range(0, N):
                    sum += image[x, y, 0] * np.exp(imagFac * ((m * x) / M + (n * y) / N))
            # print(sum)
            f = sum * quotient
            F[m + f_size, n + f_size] = f
            F[-m + f_size, -n + f_size] = f

    abs_F = np.absolute(F).reshape((2 * f_size + 1, 2 * f_size + 1, 1))

    # trimmed = trim(abs_F,1.0,np.max(abs_F))

    print(np.min(abs_F))
    print(np.max(abs_F))
    return np.log1p(abs_F)


# =================== BORDERS =======================

@jit(float64[:, :, :](float64[:, :, :], int64), nopython=True, cache=True)
def extend_with_zeros(image, border_width):
    h, w, d = image.shape
    s = border_width

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.zeros(new_shape, dtype=np.float64)
    out_image[s:h + s, s:w + s, :] = image

    return out_image


@jit(float64[:, :, :](float64[:, :, :], int64), nopython=True, cache=True)
def extend_same(image, border_width):
    # extended image:
    #
    # + ----s---- + --------w-------- + ----s---- +
    # | 111111111 | 123456789abcdefgh | hhhhhhhhh |
    # s 111111111 | 123456789abcdefgh | hhhhhhhhh s
    # | 111111111 | 123456789abcdefgh | hhhhhhhhh |
    # + --------- + ----------------- + --------- +
    # | 111111111 | 123456789abcdefgh | hhhhhhhhh |
    # | 222222222 | 2...............i | iiiiiiiii |
    # h 333333333 | 3.....IMAGE.....j | jjjjjjjjj h
    # | 444444444 | 4...............k | kkkkkkkkk |
    # | 555555555 | 56789abcdefghijkl | lllllllll |
    # + --------- + ----------------- + --------- +
    # | 555555555 | 56789abcdefghijkl | lllllllll |
    # s 555555555 | 56789abcdefghijkl | lllllllll s
    # | 555555555 | 56789abcdefghijkl | lllllllll |
    # + ----s---- + --------w-------- + ----s---- +
    #                                              \
    #                                               d
    #                                                \

    h, w, d = image.shape
    s = border_width

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.zeros(new_shape, dtype=np.float64)

    out_image[s:h + s, s:w + s, :] = image

    out_image[:s, :s, :] = np.ones((s, s, d)) * image[0, 0, :]
    out_image[:s, s + w:, :] = np.ones((s, s, d)) * image[0, -1, :]
    out_image[s + h:, :s, :] = np.ones((s, s, d)) * image[-1, 0, :]
    out_image[s + h:, s + w:, :] = np.ones((s, s, d)) * image[-1, -1, :]

    for x in range(h):
        target_row = s + x
        value_left = image[x, 0, :]
        value_right = image[x, -1, :]
        for y in range(s):
            out_image[target_row, y, :] = value_left
            out_image[target_row, y - s, :] = value_right

    for y in range(w):
        target_column = s + y
        value_up = image[0, y, :]
        value_low = image[-1, y, :]
        for x in range(s):
            out_image[x, target_column, :] = value_up
            out_image[x - s, target_column, :] = value_low

    return out_image


# ================= CONVERSION ====================

@jit(float64[:, :, :](float64[:, :, :], int64[:, :]), nopython=True, cache=True)
def label_map2label_image_avg(image, label_map):
    h, w, d = image.shape

    num_labels = np.max(label_map) + 1
    label_image = np.zeros((h, w, d), dtype=np.float64)
    avgs = np.zeros((num_labels, d), dtype=np.float64)
    pxcounter = np.zeros((num_labels), dtype=np.int64)

    ## CALCULATE AVG. COLOR / GRAY VALUE PER LABEL

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            avgs[label, :] += image[x, y, :]
            pxcounter[label] += 1

    for i in range(num_labels):
        avgs[i] /= pxcounter[i]

    ## APPLY AVG. COLOR / GRAY VALUE TO PIXELS

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            label_image[x, y, :] = avgs[label]

    return label_image


@jit(float64[:, :, :](int64[:, :]), nopython=True, cache=True)
def label_map2label_image(label_map):
    h, w = label_map.shape
    label_image = np.zeros((h, w, 3), dtype=np.float64)
    num_labels = np.max(label_map) + 1

    colors = get_saturated_colors(num_labels)

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            label_image[x, y, :] = colors[label, :]

    return label_image


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def convert_to_1channel(image):
    h, w, d = image.shape
    if d == 1:
        return np.copy(image)

    result = np.sum(image, 2) / 3.0

    return result.reshape((h, w, 1))


@jit(float64[:, :, :](float64[:, :, :], int64), nopython=True, cache=True)
def get_channel(image, channel):
    return image[:, :, channel:channel + 1:]


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def convert_to_3channel(image):
    h, w, d = image.shape
    if d == 3:
        return image

    result = np.zeros((h, w, 3), dtype=np.float64)
    result[:, :, 0] = image[:, :, 0]
    result[:, :, 1] = image[:, :, 0]
    result[:, :, 2] = image[:, :, 0]

    return result


@jit(boolean[:, :](float64[:, :, :]), nopython=True, cache=True)
def convert_to_binary(image):
    gray_image = convert_to_1channel(image)
    return gray_image[:, :, 0] != 0


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def rgb2hsv(image):
    h, w, d = image.shape
    assert d == 3
    image /= 255

    hsv_image = np.zeros((h, w, 3), dtype=np.float64)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            v_max = np.max(image[x, y, :])
            v_min = np.min(image[x, y, :])

            # HUE 0 - 360
            hue = 0.0
            if v_max > v_min:
                if r == v_max:
                    hue = 60 * (g - b) / (v_max - v_min)
                elif g == v_max:
                    hue = 120 + 60 * (b - r) / (v_max - v_min)
                elif b == v_max:
                    hue = 240 + 60 * (r - g) / (v_max - v_min)
                if hue < 0:
                    hue += 360

            # SATURATION 0 - 1
            sat = 0.0
            if v_max > 0.0:
                sat = (v_max - v_min) / v_max

            # VALUE 0 - 1
            val = v_max
            hsv_image[x, y, :] = (hue, sat, val)
    return hsv_image


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def hsv2rgb(hsv_image):
    h_, w_, d_ = hsv_image.shape
    assert d_ == 3

    rgb_image = np.zeros((h_, w_, 3), dtype=np.float64)
    for x in range(h_):
        for y in range(w_):
            h, s, v = hsv_image[x, y]
            h_i = int(h // 60) % 6
            f = h / 60 - h_i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            if h_i == 0:
                rgb = (v, t, p)
            elif h_i == 1:
                rgb = (q, v, p)
            elif h_i == 2:
                rgb = (p, v, t)
            elif h_i == 3:
                rgb = (p, q, v)
            elif h_i == 4:
                rgb = (t, p, v)
            else:
                rgb = (v, p, q)
            rgb_image[x, y, :] = rgb

    rgb_image *= 255
    return rgb_image


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def rgb2hsl(image):
    h, w, d = image.shape
    assert d == 3
    image /= 255

    hsl_image = np.zeros((h, w, 3), dtype=np.float64)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            v_max = np.max(image[x, y, :])
            v_min = np.min(image[x, y, :])

            # HUE 0 - 360
            hue = 0.0
            if v_max > v_min:
                if r == v_max:
                    hue = 60 * (g - b) / (v_max - v_min)
                elif g == v_max:
                    hue = 120 + 60 * (b - r) / (v_max - v_min)
                elif b == v_max:
                    hue = 240 + 60 * (r - g) / (v_max - v_min)
                if hue < 0:
                    hue += 360

            # SATURATION 0 - 1
            sat = 0.0
            if v_max > 0.0 and v_min < 1:
                sat = (v_max - v_min) / (1 - abs(v_max + v_min - 1))

            # LUMINANCE 0 - 1
            lum = (v_max + v_min) / 2
            hsl_image[x, y, :] = (hue, sat, lum)
    return hsl_image


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def rgb2lab(image):
    # D65 / 2 deg
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    h, w, d = image.shape
    assert d == 3

    lab_image = np.zeros((h, w, 3), dtype=np.float64)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
            Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
            Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

            frac_x = X / Xn
            frac_y = Y / Yn
            frac_z = Z / Zn

            if frac_x < 0.008856:
                root_x = 1 / 116 * (24389 / 27 * frac_x + 16)
            else:
                root_x = frac_x ** (1 / 3)

            if frac_y < 0.008856:
                root_y = 1 / 116 * (24389 / 27 * frac_y + 16)
            else:
                root_y = frac_y ** (1 / 3)

            if frac_z < 0.008856:
                root_z = 1 / 116 * (24389 / 27 * frac_z + 16)
            else:
                root_z = frac_z ** (1 / 3)

            L = 116 * root_y - 16
            a = 500 * (root_x - root_y)
            b = 200 * (root_y - root_z)

            lab_image[x, y, :] = (L, a, b)
    return lab_image


@jit(float64[:, :, :](boolean[:, :]), nopython=True, cache=True)
def binary2gray(mask):
    h, w = mask.shape
    return mask.astype(np.float64).reshape((h, w, 1)) * 255


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def rgb2pca(image):
    h, w, d = image.shape
    N = h * w
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    ### COMPUTE COVARIANCE MATRIX

    pixels = image.astype(np.float64).reshape((N, 3))
    pixels[:, 0] -= mean_r
    pixels[:, 1] -= mean_g
    pixels[:, 2] -= mean_b

    V = np.dot(pixels.T, pixels) / N

    eigvals, R = np.linalg.eig(V)

    transformed_pixels = np.dot(R.T, pixels.T)
    transformed_image = transformed_pixels.T.reshape(h, w, d)

    return normalize(transformed_image[:, :, 0:1])


### =============== HISTOGRAM =======================

@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def histogram_normalisation(image, outlier_fraction):
    h, w, d = image.shape
    N = h * w
    num_outliers = int(N * (outlier_fraction / 2))
    result = np.zeros(image.shape, dtype=np.float64)

    for z in range(d):
        channel = image[:, :, z]

        for g_low in range(256):
            if np.sum((channel < g_low)) >= num_outliers:
                break

        for g_high in range(255, -1, -1):
            if np.sum((channel > g_high)) >= num_outliers:
                break

        for x in range(h):
            for y in range(w):
                v = image[x, y, z]
                v_n = (v - g_low) * (255 / (g_high - g_low))
                result[x, y, z] = min(max(v_n, 0), 255)

    return result


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def histogram_equalization(image, alpha):
    h, w, d = image.shape
    result = np.zeros((h, w, d), dtype=np.float64)

    for z in range(d):
        channel = image[:, :, z].astype(np.ubyte)

        cumulative_dist = np.zeros(256, dtype=np.int64)
        cumulative_dist[0] = np.sum((channel == 0).astype(np.int64))
        for i in range(1, 256):
            num_i = np.sum((channel == i).astype(np.int64))

            cumulative_dist[i] = cumulative_dist[i - 1] + num_i

        mapping = cumulative_dist * 255 / cumulative_dist[-1]

        for x in range(h):
            for y in range(w):
                v = channel[x, y]
                result[x, y, z] = mapping[v]

    return result * alpha + image * (1 - alpha)


@jit(float64[:, :, :](float64[:, :, :], float64, int64), nopython=True, cache=True)
def local_histogram_equalization(image, alpha, M):
    h, w, d = image.shape
    result = np.zeros((h, w, d), dtype=np.float64)

    ### CREATE MAPPING FUNCTIONS PER BLOCK
    ext_image = extend_same(image, 3 * M)

    num_blocks_x = int(h / M) + 2
    num_blocks_y = int(w / M) + 2
    print(num_blocks_x, num_blocks_y)

    mappings = np.zeros((num_blocks_x, num_blocks_y, d, 256), dtype=np.float64)

    for x in range(num_blocks_x):
        for y in range(num_blocks_y):
            for z in range(d):
                local_channel = ext_image[3 * M + x * M - int(M / 2):3 * M + (x + 1) * M - int(M / 2),
                                3 * M + y * M - int(M / 2):3 * M + (y + 1) * M - int(M / 2), z].astype(np.ubyte)
                cumulative_dist = np.zeros(256, dtype=np.int64)
                cumulative_dist[0] = np.sum((local_channel == 0).astype(np.int64))
                for i in range(1, 256):
                    num_i = np.sum((local_channel == i).astype(np.int64))
                    cumulative_dist[i] = cumulative_dist[i - 1] + num_i
                mapping = cumulative_dist * 255 / cumulative_dist[-1]
                mappings[x, y, z, :] = mapping

    for x in range(h):
        for y in range(w):
            for z in range(d):
                v = image[x, y, z]
                xf = x / M
                yf = y / M
                x_low = int(xf)
                y_low = int(yf)
                x_high = x_low + 1
                y_high = y_low + 1
                s = (xf - x_low)
                t = (yf - y_low)
                s_ = 1.0 - s
                t_ = 1.0 - t
                f00 = mappings[x_low, y_low, z, int(v)]
                f10 = mappings[x_high, y_low, z, int(v)]
                f01 = mappings[x_low, y_high, z, int(v)]
                f11 = mappings[x_high, y_high, z, int(v)]
                result[x, y, z] = s_ * t_ * f00 + s * t_ * f10 + s_ * t * f01 + s * t * f11

    return result * alpha + image * (1 - alpha)


@jit(float64[:, :, :](float64[:, :, :]), nopython=True, cache=True)
def value_normalisation(image):
    h, w, d = image.shape
    assert d == 3
    for x in range(h):
        for y in range(w):
            max_v = np.max(image[x, y, :])
            min_v = np.min(image[x, y, :])
            spread = max_v - min_v
            # if max_v == 0:
            #     image[x, y, :] = (0, 0, 0)
            # elif spread != 0:
            #     normalized = image[x, y, :] - min_v
            #     normalized *= 255 / spread
            #
            #     weight = (spread / 255) ** 0.5
            #
            #     image[x, y, :] = weight * normalized + (1 - weight) * image[x, y, :]
            image[x, y, :] = spread

    return image


@jit(int64[:](float64[:, :, :]), nopython=True, cache=True)
def get_histogram(image):
    h, w, d = image.shape
    assert d == 1

    histogram = np.zeros((255), dtype=np.int64)
    for i in range(255):
        lb = image >= i
        ub = image < i + 1
        histogram[i] = np.sum(np.logical_and(lb, ub).astype(np.int64))

    return histogram


# ================ SUPPRESSION ====================

@jit(int64[:, :](int64[:, :], float64[:, :, :], int64), nopython=True, cache=True)
def get_n_best_3d(candidates, value_matrix, num_best):
    num_candidates = candidates.shape[0]
    if num_candidates <= num_best:
        return np.copy(candidates)

    best_candidates = np.zeros((num_best, 3), dtype=np.int64)
    values = np.zeros((num_candidates), dtype=np.float64)
    for i in range(num_candidates):
        c_x, c_y, c_z = candidates[i, :]
        values[i] = value_matrix[c_x, c_y, c_z]

    for i in range(num_best):
        best_candidate_index = np.argmax(values)
        best_candidates[i, :] = candidates[best_candidate_index, :]
        values[best_candidate_index] = -999999

    return best_candidates


@jit(int64[:, :](int64[:, :], float64[:, :, :, :], int64), nopython=True, cache=True)
def get_n_best_4d(candidates, value_matrix, num_best):
    num_candidates = candidates.shape[0]
    if num_candidates <= num_best:
        return np.copy(candidates)

    best_candidates = np.zeros((num_best, 4), dtype=np.int64)
    values = np.zeros((num_candidates), dtype=np.float64)
    for i in range(num_candidates):
        values[i] = value_matrix[candidates[i, 0], candidates[i, 1], candidates[i, 2], candidates[i, 3]]

    for i in range(num_best):
        best_candidate_index = np.argmax(values)
        best_candidates[i, :] = candidates[best_candidate_index, :]
        values[best_candidate_index] = -999999

    return best_candidates


@jit(int64[:, :](float64[:, :, :], int64), nopython=True, cache=True)
def non_max_suppression_3d(matrix, search_width):
    matrix = np.copy(matrix)
    h, w, d = matrix.shape

    num_maximas = 0
    maximas = np.zeros((len(matrix.flat), 3), dtype=np.int64)

    for x in range(search_width, h - 1 - search_width):
        for y in range(search_width, w - 1 - search_width):
            for z in range(d):
                z_low = max(z - search_width, 0)
                z_high = min(z + 1 + search_width, d)

                v = matrix[x, y, z]

                ref_area = matrix[x - search_width: x + 1 + search_width,
                           y - search_width: y + 1 + search_width,
                           z_low: z_high]

                if v < np.max(ref_area):
                    continue
                if v == np.min(ref_area):
                    continue

                maximas[num_maximas, :] = (x, y, z)
                num_maximas += 1

    return maximas[:num_maximas, :]


@jit(int64[:, :](float64[:, :, :], int64, float64), nopython=True, cache=True)
def non_max_suppression_3d_threshold(matrix, search_width, t):
    matrix = np.copy(matrix)
    h, w, d = matrix.shape

    num_maximas = 0
    maximas = np.zeros((len(matrix.flat), 3), dtype=np.int64)

    for x in range(search_width, h - 1 - search_width):
        for y in range(search_width, w - 1 - search_width):
            for z in range(d):
                z_low = max(z - search_width, 0)
                z_high = min(z + 1 + search_width, d)

                v = matrix[x, y, z]
                if v < t:
                    continue

                ref_area = matrix[x - search_width: x + 1 + search_width,
                           y - search_width: y + 1 + search_width,
                           z_low: z_high]

                if v < np.max(ref_area):
                    continue
                if v == np.min(ref_area):
                    continue

                maximas[num_maximas, :] = (x, y, z)
                num_maximas += 1

    return maximas[:num_maximas, :]


@jit(types.Tuple((int64[:], float64))(float64[:, :], float64), nopython=True, cache=True)
def get_min_coords_2d_threshold(matrix, t):
    h, w = matrix.shape

    min_coords = np.zeros((2), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            val = matrix[x, y]
            if val < t:
                t = val
                min_coords[:] = (x, y)

    return (min_coords, t)


# =============== SAMPLING ==========================

@jit(float64[:](float64[:, :, :], float64, float64), nopython=True, cache=True)
def get_sub_pixel_2d(image, x, y):
    x_low = int(x)
    x_high = x_low + 1
    y_low = int(y)
    y_high = y_low + 1

    s = x - x_low
    t = y - y_low

    s_ = 1 - s
    t_ = 1 - t

    v00 = image[x_low, y_low, :]
    v10 = image[x_high, y_low, :]
    v01 = image[x_low, y_high, :]
    v11 = image[x_high, y_high, :]

    return s_ * t_ * v00 + s * t_ * v10 + s_ * t * v01 + s * t * v11


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def rescale(image, factor):
    h, w, d = image.shape
    ext_image = extend_same(image, 1)

    h_ = int(h * factor)
    w_ = int(w * factor)
    d_ = d

    result = np.zeros((h_, w_, d_), dtype=np.float64)

    for x_ in range(h_):
        for y_ in range(w_):
            x = x_ / factor
            y = y_ / factor

            result[x_, y_, :] = get_sub_pixel_2d(ext_image, 1 + x, 1 + y)

    return result


@jit(int64[:, :, :](float64, float64, int64, float64, float64), nopython=True, cache=True)
def patch_coordinates_rotated(x, y, N, width, theta):
    coordinates = np.zeros((N, N, 2), dtype=np.int64)
    st = np.sin(theta)
    ct = np.cos(theta)
    half_N = N // 2

    for x_ in range(N):
        for y_ in range(N):
            dx = (x_ - half_N) * width / (N - 1)
            dy = (y_ - half_N) * width / (N - 1)
            coordinates[x_, y_, 0] = round(x + ct * dx - st * dy)
            coordinates[x_, y_, 1] = round(y + st * dx + ct * dy)

    return coordinates


@jit(int64[:, :, :](float64, float64, int64, float64), nopython=True, cache=True)
def patch_coordinates(x, y, N, width):
    coordinates = np.zeros((N, N, 2), dtype=np.int64)
    half_N = N // 2

    for x_ in range(N):
        for y_ in range(N):
            coordinates[x_, y_, 0] = round(x + (x_ - half_N) * width / (N - 1))
            coordinates[x_, y_, 1] = round(y + (y_ - half_N) * width / (N - 1))

    return coordinates


# ================== DRAWING ========================


@jit(void(float64[:, :, :], int64[:, :]), nopython=True)
def draw_marks(image, marks):
    for m in range(marks.shape[0]):
        x = marks[m, 0]
        y = marks[m, 1]
        image[x, y, :] = [255, 0, 0]

        for i in range(int(image.shape[0] / 100)):
            if x >= i and y >= i and x + i < image.shape[0] and y + i < image.shape[1]:
                image[x - i, y, :] = [255, 0, 0]
                image[x + i, y, :] = [255, 0, 0]
                image[x, y - i, :] = [255, 0, 0]
                image[x, y + i, :] = [255, 0, 0]


@jit(void(float64[:, :, :], float64, float64, float64, float64, float64[:]), nopython=True, cache=True)
def draw_vector(image, x0, y0, x1, y1, c):
    h, w, _ = image.shape
    if abs(x1 - x0) > abs(y1 - y0):
        if x0 > x1:
            a = x1
            x1 = x0
            x0 = a
            a = y1
            y1 = y0
            y0 = a
        range_x = x1 - x0
        range_y = y1 - y0
        for x in range(range_x):
            x_ = x0 + x
            y_ = y0 + x * range_y / range_x
            x_ = min(max(1, x_), h - 2)
            y_ = min(max(1, y_), w - 2)
            image[int(x_), int(y_), :] = c
            # image[int(x_), int(y_)-1, :] = c
            image[int(x_), int(y_) + 1, :] = c
    else:
        if y0 > y1:
            a = x1
            x1 = x0
            x0 = a
            a = y1
            y1 = y0
            y0 = a
        range_x = x1 - x0
        range_y = y1 - y0
        for y in range(range_y):
            y_ = y0 + y
            x_ = x0 + y * range_x / range_y
            x_ = min(max(1, x_), h - 2)
            y_ = min(max(1, y_), w - 2)
            image[int(x_), int(y_), :] = c
            # image[int(x_)-1, int(y_), :] = c
            image[int(x_) + 1, int(y_), :] = c


@jit(void(float64[:, :, :], float64[:, :]), nopython=True, cache=True)
def draw_vectors(image, vectors):
    for i in range(vectors.shape[0]):
        x0, y0, x1, y1 = vectors[i, :]
        c = get_random_color()
        draw_vector(image, x0, y0, x1, y1, c)


@jit(void(float64[:, :, :], float64[:, :]), nopython=True, cache=True)
def draw_red_vectors(image, vectors):
    for i in range(vectors.shape[0]):
        x0, y0, x1, y1 = vectors[i, :]
        c = np.array([255, 0, 0], dtype=np.float64)
        draw_vector(image, x0, y0, x1, y1, c)


@jit(void(float64[:, :, :], float64, float64, float64), nopython=True, cache=True)
def draw_circle(image, x, y, r):
    color = np.array([255, 0, 0], dtype=np.float64)

    x0 = x
    y0 = int(y + r)

    for deg in range(10, 370, 10):
        rad = np.deg2rad(deg)
        dx = np.sin(rad) * r
        dy = np.cos(rad) * r

        x1 = int(x + dx)
        y1 = int(y + dy)

        draw_vector(image, x0, y0, x1, y1, color)
        x0 = x1
        y0 = y1


@jit(void(float64[:, :, :], float64[:, :]), nopython=True, cache=True)
def draw_circles(image, circles):
    for c in range(circles.shape[0]):
        x, y, r = circles[c, :]
        draw_circle(image, x, y, r)


@jit(void(float64[:, :, :], float64[:, :]), nopython=True, cache=True)
def draw_lines(image, lines):
    for i in range(lines.shape[0]):
        d, omega, _ = lines[i, :]
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        if not cos_omega == 0.0:
            for x in range(image.shape[0]):
                y = (d - sin_omega * x) / cos_omega
                if y >= 0 and y < image.shape[1]:
                    image[int(x), int(y), :] = (255, 0, 0)
        if not sin_omega == 0.0:
            for y in range(image.shape[1]):
                x = (d - cos_omega * y) / sin_omega
                if x >= 0 and x < image.shape[0]:
                    image[int(x), int(y), :] = (255, 0, 0)


def plot(image, id):
    if image is None:
        return
    plt.figure(id)

    if image.shape[2] == 1:
        im = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.ubyte)
        im[:, :, [0]] = image
        im[:, :, [1]] = image
        im[:, :, [2]] = image
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(image.astype(np.ubyte), cmap=plt.cm.gray)

    plt.show()


def plot_hist(hist, name=''):
    plt.figure(name)
    plt.plot(hist, label='Gray Values', color='gray')
    plt.legend()
    plt.show()


def plot_hist3(gray, red, green, blue, name=''):
    plt.figure(name)
    plt.plot(red, label='Red', color='red')
    plt.plot(green, label='Green', color='green')
    plt.plot(blue, label='Blue', color='blue')
    plt.plot(gray, label='Gray', color='gray')
    plt.legend()
    plt.show()
