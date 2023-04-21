import numpy as np
import tools
from numba import jit, float64, int64
from scipy import special

GROUPNAME = 'Convolution'

BLOCK = 'Block'
GAUSS = 'Gauss'
BINOMIAL = 'Binomial'

SIMPLE_FIRST_DERIVATIVE = 'Simple First Derivative'
SOBEL = 'Sobel'
DERIVATIVE_OF_GAUSSIAN = 'Derivative of Gaussian'

SIMPLE_SECOND_DERIVATIVE = 'Simple Second Derivative'
LAPLACIAN_OF_GAUSSIAN = 'Laplacian of Gaussian'
SECOND_DERIVATIVE_OF_GAUSSIAN = 'Second Derivative of Gaussian'

MEDIAN = 'Median'
MAXIMUM = 'Maximum'
MINIMUM = 'Minimum'

DIFFERENCE_OF_GAUSSIAN = 'Difference of Gaussian'
SHARPEN = 'Sharpen'

OPS = (BLOCK, GAUSS, BINOMIAL, SIMPLE_FIRST_DERIVATIVE, SOBEL, DERIVATIVE_OF_GAUSSIAN, SIMPLE_SECOND_DERIVATIVE,
       LAPLACIAN_OF_GAUSSIAN, SECOND_DERIVATIVE_OF_GAUSSIAN, MEDIAN, MAXIMUM, MINIMUM, DIFFERENCE_OF_GAUSSIAN,
       SHARPEN)

# CONSTANTS:

ZERO = 0
SAME = 1
VALID = 2

MED = 0
MIN = 1
MAX = 2


def apply(image, operation, p1, p2, p3, p4):
    # ============= PARSE PADDING ============
    if p3 == 'same':
        pad = SAME
    elif p3 == 'valid':
        pad = VALID
    elif p3 == 'zero':
        pad = ZERO
    else:
        raise ValueError

    # ============= APPLY FILTER ===============

    if operation == BLOCK:
        N = int(float(p1))
        filter_mat = box_filter(N)
        return fast_sw_convolution(image, filter_mat, pad)

    elif operation == BINOMIAL:
        N = int(float(p1))
        filter_mat = binomial_filter(N)
        return fast_sw_convolution(image, filter_mat, pad)

    elif operation == GAUSS:
        sigma = float(p1)
        filter_mat = gauss_filter(sigma)
        return fast_sw_convolution(image, filter_mat, pad)

    elif operation == SIMPLE_FIRST_DERIVATIVE:
        filter_mat_x = simple_first_derivative(0)
        filter_mat_y = simple_first_derivative(1)
        if p4 == 'x':
            return tools.normalize(fast_sw_convolution(image, filter_mat_x, pad))
        elif p4 == 'y':
            return tools.normalize(fast_sw_convolution(image, filter_mat_y, pad))
        elif p4 == 'amplitude' or p4 == 'amp' or p4 == 'a':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            return tools.normalize(a)
        elif p4 == 'direction' or p4 == 'dir' or p4 == 'd':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.arctan2(dy, dx)
            return (a * 255 / (2 * np.pi))
        elif p4 == 'color' or p4 == 'col' or p4 == 'c':
            image = tools.convert_to_1channel(image)
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            a /= np.max(a)
            d = np.arctan2(dy, dx)
            d *= 180 / np.pi
            d += (d < 0).astype(np.float64) * 360

            hsv_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.float64)
            hsv_image[:, :, 0:1:] = d
            hsv_image[:, :, 1:2:] = a
            rgb_image = tools.hsv2rgb(hsv_image)
            return rgb_image
        else:
            raise ValueError

    elif operation == SOBEL:
        filter_mat_x = sobel(0)
        filter_mat_y = sobel(1)
        if p4 == 'x':
            return tools.normalize(fast_sw_convolution(image, filter_mat_x, pad))
        elif p4 == 'y':
            return tools.normalize(fast_sw_convolution(image, filter_mat_y, pad))
        elif p4 == 'amplitude' or p4 == 'amp' or p4 == 'a':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            return tools.normalize(a)
        elif p4 == 'direction' or p4 == 'dir' or p4 == 'd':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.arctan2(dy, dx)
            return (a * 255 / (2 * np.pi))
        elif p4 == 'color' or p4 == 'col' or p4 == 'c':
            image = tools.convert_to_1channel(image)
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            a /= np.max(a)
            d = np.arctan2(dy, dx)
            d *= 180 / np.pi
            d += (d < 0).astype(np.float64) * 360

            hsv_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.float64)
            hsv_image[:, :, 0:1:] = d
            hsv_image[:, :, 1:2:] = a
            rgb_image = tools.hsv2rgb(hsv_image)
            return rgb_image
        else:
            raise ValueError

    elif operation == DERIVATIVE_OF_GAUSSIAN:
        sigma = float(p1)
        filter_mat_x = derivative_of_gaussian(sigma, 0)
        filter_mat_y = derivative_of_gaussian(sigma, 1)
        if p4 == 'x':
            return tools.normalize(fast_sw_convolution(image, filter_mat_x, pad))
        elif p4 == 'y':
            return tools.normalize(fast_sw_convolution(image, filter_mat_y, pad))
        elif p4 == 'amplitude' or p4 == 'amp' or p4 == 'a':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            return tools.normalize(a)
        elif p4 == 'direction' or p4 == 'dir' or p4 == 'd':
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            d = np.arctan2(dy, dx)
            return (d * 255 / (2 * np.pi))
        elif p4 == 'color' or p4 == 'col' or p4 == 'c':
            image = tools.convert_to_1channel(image)
            dx = fast_sw_convolution(image, filter_mat_x, pad)
            dy = fast_sw_convolution(image, filter_mat_y, pad)
            a = np.sqrt(np.square(dx) + np.square(dy))
            a /= np.max(a)
            d = np.arctan2(dy, dx)
            d *= 180 / np.pi
            d += (d < 0).astype(np.float64) * 360

            hsv_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.float64)
            hsv_image[:, :, 0:1:] = d
            hsv_image[:, :, 1:2:] = a
            rgb_image = tools.hsv2rgb(hsv_image)
            return rgb_image
        else:
            raise ValueError

    elif operation == SIMPLE_SECOND_DERIVATIVE:
        filter_mat_xx = simple_second_derivative(0)
        filter_mat_yy = simple_second_derivative(1)
        filter_mat_xy = simple_second_derivative(2)
        if p4 == 'xx':
            return tools.normalize(fast_sw_convolution(image, filter_mat_xx, pad))
        elif p4 == 'yy':
            return tools.normalize(fast_sw_convolution(image, filter_mat_yy, pad))
        elif p4 == 'xy':
            return tools.normalize(fast_sw_convolution(image, filter_mat_xy, pad))
        elif p4 == 'det':
            g_xx = fast_sw_convolution(image, filter_mat_xx, pad)
            g_yy = fast_sw_convolution(image, filter_mat_yy, pad)
            g_xy = fast_sw_convolution(image, filter_mat_xy, pad)
            det = g_xx * g_yy - g_xy * g_xy
            return tools.normalize(det)
        else:
            raise ValueError

    elif operation == SECOND_DERIVATIVE_OF_GAUSSIAN:
        sigma = float(p1)
        filter_mat_xx = second_derivative_of_gaussian(sigma, 0)
        filter_mat_yy = second_derivative_of_gaussian(sigma, 1)
        filter_mat_xy = second_derivative_of_gaussian(sigma, 2)
        if p4 == 'xx':
            return tools.normalize(fast_sw_convolution(image, filter_mat_xx, pad))
        elif p4 == 'yy':
            return tools.normalize(fast_sw_convolution(image, filter_mat_yy, pad))
        elif p4 == 'xy':
            return tools.normalize(fast_sw_convolution(image, filter_mat_xy, pad))
        elif p4 == 'det':
            g_xx = fast_sw_convolution(image, filter_mat_xx, pad)
            g_yy = fast_sw_convolution(image, filter_mat_yy, pad)
            g_xy = fast_sw_convolution(image, filter_mat_xy, pad)
            det = g_xx * g_yy - g_xy * g_xy
            return tools.normalize(det)
        else:
            raise ValueError

    elif operation == LAPLACIAN_OF_GAUSSIAN:
        sigma = float(p1)
        filter_mat = laplacian_of_gaussian(sigma)
        return tools.normalize(fast_sw_convolution(image, filter_mat, pad))

    elif operation == MEDIAN:
        N = int(float(p1))
        return sliding_window_nonlin_convolution(image, MED, N, pad)

    elif operation == MINIMUM:
        N = int(float(p1))
        return sliding_window_nonlin_convolution(image, MIN, N, pad)

    elif operation == MAXIMUM:
        N = int(float(p1))
        return sliding_window_nonlin_convolution(image, MAX, N, pad)

    elif operation == DIFFERENCE_OF_GAUSSIAN:
        sigma_high = float(p1)
        filter_mat_high = gauss_filter(sigma_high)
        gauss_high = fast_sw_convolution(image, filter_mat_high, pad)

        sigma_low = float(p2)
        filter_mat_low = gauss_filter(sigma_low)
        gauss_low = fast_sw_convolution(image, filter_mat_low, pad)

        return tools.normalize(gauss_high - gauss_low)

    elif operation == SHARPEN:
        sigma = float(p1)
        gamma = float(p2)

        return sharpen(image, sigma, gamma)


# =================== BLURRING ======================

@jit(float64[:, :](int64), nopython=True, cache=True)
def box_filter(N):
    if N % 2 == 0:
        N += 1
    filter = np.ones((N, N), dtype=np.float64) / (N * N)
    return filter


def binomial_filter(N):
    f = 1.0 / ((2 ** (N - 1)))
    v = np.zeros((N, 1), dtype=np.float64)
    for i in range(N):
        v[i] = special.binom(N - 1, i)
    v *= f
    filter = v.dot(v.T)
    return filter.astype(np.float)


@jit(float64[:, :](float64, int64), nopython=True, cache=True)
def gauss_filter_N(sigma, N):
    filter = np.zeros((N, N), dtype=np.float64)
    offset = N / 2 - 0.5
    two_sigma_sq = 2 * sigma * sigma
    for x_ in range(N):
        for y_ in range(N):
            x = x_ - offset
            y = y_ - offset
            G_sigma = 1.0 / (np.pi * two_sigma_sq) * np.e ** (-(x * x + y * y) / (two_sigma_sq))
            filter[x_, y_] = G_sigma
    return filter


@jit(float64[:, :](float64), nopython=True, cache=True)
def gauss_filter(sigma):
    N = int(6 * sigma)
    if N % 2 == 0:
        N += 1
    return gauss_filter_N(sigma, N)


# ================= 1st DERIVATIVE ==================


@jit(float64[:, :](int64), nopython=True, cache=True)
def simple_first_derivative(axis=0):
    h = (1 / 2) * np.array(((0, 1, 0), (0, 0, 0), (0, -1, 0)), dtype=np.float64)
    if axis == 1:
        h = h.T
    return h


@jit(float64[:, :](int64), nopython=True, cache=True)
def sobel(axis=0):
    h = (1 / 8) * np.array(((1, 2, 1), (0, 0, 0), (-1, -2, -1)), dtype=np.float64)
    if axis == 1:
        h = h.T
    return h


@jit(float64[:, :](float64, int64), nopython=True, cache=True)
def derivative_of_gaussian(sigma, axis=0):
    N = int(6 * sigma)
    if N % 2 == 0:
        N += 1
    filter = np.zeros((N, N), dtype=np.float64)
    offset = int(N / 2)
    two_sigma_sq = 2 * sigma * sigma
    two_pi_sigma_4 = 2 * np.pi * sigma ** 4

    if axis == 0:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = - x / two_pi_sigma_4 *  np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    else:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = - y / two_pi_sigma_4 * \
                          np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    return filter


# ================= 2nd DERIVATIVE ==================

@jit(float64[:, :](int64), nopython=True, cache=True)
def simple_second_derivative(axis=0):
    if axis == 0 or axis == 1:
        h = np.array(((0, 1, 0), (0, -2, 0), (0, 1, 0)), dtype=np.float64)
        if axis == 1:
            h = h.T
    elif axis == 2:  # g_xy
        h = (1 / 4) * np.array(((1, 0, -1), (0, 0, 0), (-1, 0, 1)), dtype=np.float64)
    return h


@jit(float64[:, :](float64, int64), nopython=True, cache=True)
def second_derivative_of_gaussian(sigma, axis=0):
    N = int(6 * sigma)
    if N % 2 == 0:
        N += 1
    filter = np.zeros((N, N), dtype=np.float64)
    offset = int(N / 2)
    two_sigma_sq = 2 * sigma * sigma
    two_pi_sigma_6 = 2 * np.pi * sigma ** 6

    if axis == 0:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = (x - sigma) * (x + sigma) / two_pi_sigma_6 * \
                          np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    elif axis == 1:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = (y - sigma) * (y + sigma) / two_pi_sigma_6 * \
                          np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    else:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = x * y / two_pi_sigma_6 * \
                          np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma

    return filter


@jit(float64[:, :](float64), nopython=True, cache=True)
def laplacian_of_gaussian(sigma):
    N = int(6 * sigma)
    if N % 2 == 0:
        N += 1
    filter = np.empty((N, N), dtype=np.float64)
    offset = int(N / 2)

    f0 = -1 / (np.pi * sigma ** 4)
    two_sigma_sq = 2 * sigma ** 2

    for x_ in range(N):
        for y_ in range(N):
            x = x_ - offset
            y = y_ - offset
            xxyy = x * x + y * y
            G_sigma = f0 * (1 - xxyy / two_sigma_sq) * np.exp(-xxyy / two_sigma_sq)
            filter[x_, y_] = G_sigma

    return filter


# ================= CONVOLUTION ==================

@jit(float64[:, :, :](float64[:, :, :], float64[:, :]), nopython=True)
def sw_convolution(image, filter):
    fs = filter.shape[0]
    hfs = int(fs / 2)
    h, w, d = image.shape

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    for y_ in range(fs):
                        v += image[x + x_, y + y_, z] * filter[x_, y_]

                out_image[x, y, z] = v

    return out_image


@jit(float64[:, :, :](float64[:, :, :], float64[:], float64[:]), nopython=True)
def fast_sw_convolution_sv(image, u0, v0):
    fs = u0.shape[0]
    hfs = int(fs / 2)

    h, w, d = image.shape

    mid_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)
    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    v += image[x + x_, y + hfs, z] * u0[x_]
                mid_image[x, y, z] = v

    mid_image = tools.extend_same(mid_image, hfs)

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for y_ in range(fs):
                    v += mid_image[x + hfs, y + y_, z] * v0[y_]
                out_image[x, y, z] = v

    return out_image


@jit(float64[:, :, :](float64[:, :, :], float64[:, :], int64), nopython=True)
def fast_sw_convolution(in_image, filter, padding):
    fs = filter.shape[0]
    hfs = int(fs / 2)

    if padding == SAME:
        image = tools.extend_same(in_image, hfs)
    elif padding == ZERO:
        image = tools.extend_with_zeros(in_image, hfs)
    else:
        image = np.copy(in_image)

    u, s, vh = np.linalg.svd(filter, True)
    if s[1] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        return fast_sw_convolution_sv(image, u0, v0)
    if s[2] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        s1_root = np.sqrt(s[1])
        u1 = u[:, 1] * s1_root
        v1 = vh[1, :] * s1_root
        return fast_sw_convolution_sv(image, u0, v0) + \
               fast_sw_convolution_sv(image, u1, v1)
    else:
        print('fallback to sw_convolution')
        return sw_convolution(image, filter)


@jit(float64[:, :, :](float64[:, :, :], int64, int64, int64), nopython=True, cache=True)
def sliding_window_nonlin_convolution(in_image, operation, N, padding):
    if N % 2 == 0:
        N += 1
    hfs = int(N / 2)

    if padding == SAME:
        image = tools.extend_same(in_image, hfs)
    elif padding == ZERO:
        image = tools.extend_with_zeros(in_image, hfs)
    else:
        image = np.copy(in_image)

    h, w, d = image.shape
    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float64)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                area = image[x:x + N, y:y + N, z]
                if operation == MED:
                    out_image[x, y, z] = np.median(area)
                elif operation == MAX:
                    out_image[x, y, z] = np.max(area)
                elif operation == MIN:
                    out_image[x, y, z] = np.min(area)

    return out_image


@jit(float64(float64[:, :, :], float64[:, :], int64, int64, int64), nopython=True, cache=True)
def convolve_at(image, filter, x, y, z):
    h, w, _ = image.shape
    filter_size = filter.shape[0]
    half_ks = int(filter_size / 2)
    x_low = x - half_ks
    y_low = y - half_ks

    v = 0.0
    for x_ in range(filter_size):
        for y_ in range(filter_size):
            xi = min(max(0, x_low + x_), h - 1)
            yi = min(max(0, y_low + y_), w - 1)
            v += image[xi, yi, z] * filter[x_, y_]

    return v


# ================= OTHERS ======================


@jit(float64[:, :, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def sharpen(image, sigma, gamma):
    filter_mat = gauss_filter(sigma)

    gauss = fast_sw_convolution(image, filter_mat, SAME)
    sharpened = image + gamma * (image - gauss)
    sharpened = tools.trim(sharpened, 0, 255)

    return sharpened
