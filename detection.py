import numpy as np
import tools
from numba import jit, float64, int64, types
from scipy import special
import convolution
import matplotlib.pyplot as plt
import math

GROUPNAME = 'Detection'

LOG_DETECTOR = 'LoG Detector'
BEAUDET = 'Beaudet'
HARRIS = 'Harris'
SCALE_ADAPTED_HARRIS = 'Scale Adapted Harris'
FOERSTNER = 'Foerstner'
DIFFERENCE_OF_GAUSSIAN = 'Difference of Gaussian'
EDGE_PIXELS = 'Edge Pixels'
HOUGH_LINES = 'Hough Lines'

OPS = (BEAUDET, HARRIS, SCALE_ADAPTED_HARRIS, FOERSTNER, LOG_DETECTOR,
       DIFFERENCE_OF_GAUSSIAN, EDGE_PIXELS, HOUGH_LINES)


def apply(in_image, filter, p1, p2, p3, p4):
    gray_in_image = tools.convert_to_1channel(in_image)

    if filter == BEAUDET:
        sigma = float(p1)
        num_best = int(float(p4))

        maxs = beaudet(gray_in_image, sigma, num_best)
        result = tools.convert_to_3channel(in_image)
        tools.draw_marks(result, maxs)
        return result

    if filter == HARRIS:
        N = int(float(p1))
        kappa = float(p2)
        r_min = float(p3)
        num_best = int(float(p4))

        maxs = harris(gray_in_image, N, kappa, r_min, num_best)
        result = tools.convert_to_3channel(in_image)
        tools.draw_marks(result, maxs)
        return result

    if filter == SCALE_ADAPTED_HARRIS:
        sigma = float(p1)
        kappa = float(p2)
        r_min = int(float(p3))
        num_best = int(float(p4))

        maxs = scale_adapted_harris(gray_in_image, sigma, kappa, r_min, num_best)
        result = tools.convert_to_3channel(in_image)
        tools.draw_marks(result, maxs)
        return result

    if filter == FOERSTNER:
        sigma = float(p1)
        omega_min = float(p2)
        q_min = float(p3)
        num_best = int(float(p4))

        maxs = foerstner(gray_in_image, sigma, omega_min, q_min, num_best)
        result = tools.convert_to_3channel(in_image)
        tools.draw_marks(result, maxs)
        return result

    if filter == LOG_DETECTOR:
        sigma_min = float(p1)
        sigma_max = float(p2)
        steps = int(float(p3))
        num_best = int(float(p4))

        blobs = log_detector(gray_in_image, sigma_min, sigma_max, steps, num_best)
        result = tools.convert_to_3channel(in_image)
        tools.draw_circles(result, blobs)
        return result

    if filter == DIFFERENCE_OF_GAUSSIAN:
        octaves = int(float(p1))
        num_per_octave = int(float(p2))
        best_pct = float(p3)
        threshold = float(p4)

        features = difference_of_gaussian(gray_in_image, octaves, num_per_octave, threshold, best_pct)

        circles = np.copy(features)
        circles[:, 2] *= np.sqrt(2)

        result = tools.convert_to_3channel(in_image)
        tools.draw_circles(result, circles)
        return result

    if filter == EDGE_PIXELS:
        sigma = float(p1)
        t = float(p2)
        return edge_pixels(gray_in_image, sigma, t)

    if filter == HOUGH_LINES:
        sigma = float(p1)
        show = int(float(p2))
        t_amp = float(p3)
        n_best = int(float(p4))

        if show == 2:
            hough_space = hough_lines_nice(in_image, sigma)
            return tools.normalize(hough_space)

        show_hough_space = show == 1

        lines, hough_space = hough_lines(gray_in_image, sigma, t_amp, n_best, show_hough_space)

        if show_hough_space:
            return hough_space
        else:
            result = tools.convert_to_3channel(in_image)
            tools.draw_lines(result, lines)
            return result


@jit(int64[:, :](float64[:, :, :], float64, int64), nopython=True, cache=True)
def beaudet(image, sigma, num_best):
    filter_mat_xx = convolution.second_derivative_of_gaussian(sigma, 0)
    filter_mat_yy = convolution.second_derivative_of_gaussian(sigma, 1)
    filter_mat_xy = convolution.second_derivative_of_gaussian(sigma, 2)

    g_xx = convolution.fast_sw_convolution(image, filter_mat_xx, convolution.SAME)
    g_yy = convolution.fast_sw_convolution(image, filter_mat_yy, convolution.SAME)
    g_xy = convolution.fast_sw_convolution(image, filter_mat_xy, convolution.SAME)

    det = g_xx * g_yy - g_xy * g_xy

    maximums = tools.non_max_suppression_3d(det, 1)
    n_best = tools.get_n_best_3d(maximums, det, num_best)
    return n_best


@jit(int64[:, :](float64[:, :, :], int64, float64, float64, int64), nopython=True, cache=True)
def harris(image, N, kappa, r_min, num_best):
    filter_gx = convolution.simple_first_derivative(0)
    filter_gy = convolution.simple_first_derivative(1)

    g_x = convolution.fast_sw_convolution(image, filter_gx, convolution.SAME)
    g_y = convolution.fast_sw_convolution(image, filter_gy, convolution.SAME)

    g_x_sq = g_x * g_x
    g_y_sq = g_y * g_y
    g_x_y = g_x * g_y

    weighting_filter = convolution.box_filter(N)

    mean_g_x_sq = convolution.fast_sw_convolution(g_x_sq, weighting_filter, convolution.SAME)
    mean_g_y_sq = convolution.fast_sw_convolution(g_y_sq, weighting_filter, convolution.SAME)
    mean_g_x_y = convolution.fast_sw_convolution(g_x_y, weighting_filter, convolution.SAME)

    r_map = np.zeros(image.shape, dtype=np.float64)

    M = np.zeros((2, 2), dtype=np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            M[0, 0] = mean_g_x_sq[x, y, 0]
            M[0, 1] = mean_g_x_y[x, y, 0]
            M[1, 0] = mean_g_x_y[x, y, 0]
            M[1, 1] = mean_g_y_sq[x, y, 0]

            r = np.linalg.det(M) - kappa * np.square(np.trace(M))
            r_map[x, y, 0] = r

    maximums = tools.non_max_suppression_3d_threshold(r_map, 1, r_min)
    n_best = tools.get_n_best_3d(maximums, r_map, num_best)
    return n_best


@jit(int64[:, :](float64[:, :, :], float64, float64, float64, int64), nopython=True, cache=True)
def scale_adapted_harris(image, sigma, kappa, r_min, num_best):
    filter_gx = convolution.derivative_of_gaussian(sigma, 0)
    filter_gy = convolution.derivative_of_gaussian(sigma, 1)

    g_x = convolution.fast_sw_convolution(image, filter_gx, convolution.SAME)
    g_y = convolution.fast_sw_convolution(image, filter_gy, convolution.SAME)

    g_xx = g_x * g_x
    g_yy = g_y * g_y
    g_xy = g_x * g_y

    integration_scale = 2 * sigma

    filter_mean = convolution.gauss_filter(integration_scale)

    mean_gxx = convolution.fast_sw_convolution(g_xx, filter_mean, convolution.SAME)
    mean_gyy = convolution.fast_sw_convolution(g_yy, filter_mean, convolution.SAME)
    mean_gxy = convolution.fast_sw_convolution(g_xy, filter_mean, convolution.SAME)

    r_map = np.zeros(image.shape, dtype=np.float64)

    M = np.zeros((2, 2), dtype=np.float64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            M[0, 0] = mean_gxx[x, y, 0]
            M[0, 1] = mean_gxy[x, y, 0]
            M[1, 0] = mean_gxy[x, y, 0]
            M[1, 1] = mean_gyy[x, y, 0]

            r = np.linalg.det(M) - kappa * np.square(np.trace(M))
            r_map[x, y, 0] = r

    maximums = tools.non_max_suppression_3d_threshold(r_map, 1, r_min)
    print(maximums.shape[0])
    n_best = tools.get_n_best_3d(maximums, r_map, num_best)
    print(n_best.shape[0])
    return n_best


@jit(int64[:, :](float64[:, :, :], float64, float64, float64, int64), nopython=True, cache=True)
def foerstner(image, sigma, omega_min, q_min, num_best):
    filter_gx = convolution.derivative_of_gaussian(sigma, 0)
    filter_gy = convolution.derivative_of_gaussian(sigma, 1)

    g_x = convolution.fast_sw_convolution(image, filter_gx, convolution.SAME)
    g_y = convolution.fast_sw_convolution(image, filter_gy, convolution.SAME)

    g_xx = g_x * g_x
    g_yy = g_y * g_y
    g_xy = g_x * g_y

    integration_scale = 2 * sigma

    filter_mean = convolution.gauss_filter(integration_scale)

    mean_gxx = convolution.fast_sw_convolution(g_xx, filter_mean, convolution.SAME)
    mean_gyy = convolution.fast_sw_convolution(g_yy, filter_mean, convolution.SAME)
    mean_gxy = convolution.fast_sw_convolution(g_xy, filter_mean, convolution.SAME)

    omega_map = np.zeros(image.shape, dtype=np.float64)
    # omega_map_full = np.zeros(image.shape, dtype=np.float64)
    # q_map_full = np.zeros(image.shape, dtype=np.float64)
    M = np.zeros((2, 2), dtype=np.float64)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            M[0, 0] = mean_gxx[x, y, 0]
            M[0, 1] = mean_gxy[x, y, 0]
            M[1, 0] = mean_gxy[x, y, 0]
            M[1, 1] = mean_gyy[x, y, 0]

            det_M = np.linalg.det(M)
            trace_M = np.trace(M)

            if trace_M == 0:
                continue

            omega = det_M / trace_M
            q = 4 * det_M / (trace_M ** 2)

            # omega_map_full[x,y,0] = omega
            # q_map_full[x,y,0] = q

            if q > q_min:
                omega_map[x, y, 0] = omega

    maximums = tools.non_max_suppression_3d_threshold(omega_map, 1, omega_min)
    n_best = tools.get_n_best_3d(maximums, omega_map, num_best)
    return n_best


@jit(float64[:, :](float64[:, :, :], float64, float64, int64, int64), nopython=True, cache=False)
def log_detector(image, s_min, s_max, steps, num_best):
    abs_scale_space = np.zeros((image.shape[0], image.shape[1], steps), dtype=np.float64)
    sigmas = np.linspace(s_min, s_max, steps)

    for i in range(steps):
        sigma = sigmas[i]
        filter_mat = convolution.laplacian_of_gaussian(sigma)
        log = convolution.fast_sw_convolution(image, filter_mat, convolution.SAME)
        n_log = log * (sigma * sigma)
        abs_scale_space[:, :, i] = np.abs(n_log[:, :, 0])

    candidates = tools.non_max_suppression_3d(abs_scale_space, 1)
    candidates = tools.get_n_best_3d(candidates, abs_scale_space, num_best)

    circles = np.copy(candidates).astype(np.float64)

    for c in range(circles.shape[0]):
        circles[c, 2] = sigmas[candidates[c, 2]] * np.sqrt(2)

    return circles


@jit(float64[:, :](float64[:, :, :], int64, int64, float64, int64), nopython=True, cache=True)
def difference_of_gaussian(image, nr_octaves, num_gauss, min_abs_dog, best_pct):
    r_max = 10
    t_h = ((r_max + 1) ** 2) / r_max
    k = 2 ** (1 / (num_gauss - 2))

    sigmas = []
    for i in range(num_gauss):
        sigmas.append(k ** i)

    h0, w0, _ = image.shape
    gaussians = np.zeros((nr_octaves, h0, w0, num_gauss), dtype=np.float64)
    differences = np.zeros((nr_octaves, h0, w0, num_gauss - 1), dtype=np.float64)

    ## COMPUTE GAUSSIANS

    for i in range(nr_octaves):
        sampled_image = tools.rescale(image, 1/(2 ** i))
        hi, wi, _ = sampled_image.shape
        for j in range(num_gauss):
            gauss_filter = convolution.gauss_filter(sigmas[j])
            g = convolution.fast_sw_convolution(sampled_image, gauss_filter, convolution.SAME)
            gaussians[i, :hi, :wi, j] = g[:, :, 0]

    ## COMPUTE DIFFERENCES OF GAUSSIANS

    for i in range(nr_octaves):
        hi = h0 / (2 ** i)
        wi = w0 / (2 ** i)
        h, w, d = gaussians[i].shape
        for j in range(d - 1):
            diff = gaussians[i, :hi, :wi, j + 1] - gaussians[i, :hi, :wi, j]
            differences[i, :hi, :wi, j] = diff[:, :]

    ## REMOVE LOW RESPONSE AND EDGE POINTS

    ssd_xx_filter = convolution.simple_second_derivative(0)
    ssd_yy_filter = convolution.simple_second_derivative(1)
    ssd_xy_filter = convolution.simple_second_derivative(2)

    candidates = np.zeros((h0 * w0 * num_gauss * nr_octaves, 4), dtype=np.int64)
    candidate_counter = 0
    for i in range(nr_octaves):
        hi = h0 / (2 ** i)
        wi = w0 / (2 ** i)
        differences_i = differences[i, :hi, :wi, :]
        candidates_i = tools.non_max_suppression_3d(np.abs(differences_i), 1)  # [c] = x,y,z

        for c in range(candidates_i.shape[0]):
            x, y, z = candidates_i[c, :]
            v = differences[i, x, y, z]
            if abs(v) < min_abs_dog:
                continue
            dxx = convolution.convolve_at(differences_i, ssd_xx_filter, x, y, z)
            dyy = convolution.convolve_at(differences_i, ssd_yy_filter, x, y, z)
            dxy = convolution.convolve_at(differences_i, ssd_xy_filter, x, y, z)

            H = np.array(((dxx, dxy), (dxy, dyy)))
            det_H = np.linalg.det(H)

            if det_H > 0:
                trace_H = np.trace(H)
                if np.square(trace_H) / det_H < t_h:
                    candidates[candidate_counter, :] = (i, x, y, z)
                    candidate_counter += 1

    candidates = candidates[:candidate_counter, :]

    ## GET N BEST CANDIDATES

    num_best = int(candidate_counter * (best_pct/100))
    candidates = tools.get_n_best_4d(candidates, np.abs(differences), num_best)
    num_candidates = candidates.shape[0]

    ## CONVERT ORIGINAL X,Y,SIGMA  (o,x,y,z) to (xf,yf,sf)

    candidates_f = np.zeros((num_candidates, 3), dtype=np.float64)
    for c in range(num_candidates):
        o, x, y, z = candidates[c, :]
        xf = x * (2.0 ** o)
        yf = y * (2.0 ** o)
        sf = (2 ** o) * sigmas[z]
        candidates_f[c, :] = (xf, yf, sf)

    ## REFINEMENT

    step = 0.5
    num_candidates = candidates_f.shape[0]
    max_used_it = 0
    for c in range(num_candidates):
        xf, yf, sf = candidates_f[c, :]
        x = int(xf)
        y = int(yf)

        log_filter = convolution.laplacian_of_gaussian(sf)
        v = sf ** 2 * abs(convolution.convolve_at(image, log_filter, x, y, 0))

        for it in range(100):
            if y > 0:
                v_left = sf ** 2 * abs(convolution.convolve_at(image, log_filter, x, y - 1, 0))
                if v_left > v:
                    y -= 1
                    v = v_left
                    continue

            if y < w0 - 1:
                v_right = sf ** 2 * abs(convolution.convolve_at(image, log_filter, x, y + 1, 0))
                if v_right > v:
                    y += 1
                    v = v_right
                    continue

            if x > 0:
                v_up = sf ** 2 * abs(convolution.convolve_at(image, log_filter, x - 1, y, 0))
                if v_up > v:
                    x -= 1
                    v = v_up
                    continue

            if x < h0 - 1:
                v_down = sf ** 2 * abs(convolution.convolve_at(image, log_filter, x + 1, y, 0))
                if v_down > v:
                    x += 1
                    v = v_down
                    continue

            log_filter_high = convolution.laplacian_of_gaussian(sf + step)
            v_high = (sf + step) ** 2 * abs(convolution.convolve_at(image, log_filter_high, x, y, 0))
            if v_high > v:
                sf += step
                log_filter = convolution.laplacian_of_gaussian(sf)
                continue

            if sf - step >= 0.4:
                log_filter_low = convolution.laplacian_of_gaussian(sf - step)
                v_low = (sf - step) ** 2 * abs(convolution.convolve_at(image, log_filter_low, x, y, 0))
                if v_low > v:
                    sf -= step
                    log_filter = convolution.laplacian_of_gaussian(sf)
                    continue

            if it > max_used_it:
                max_used_it = it
            break

        candidates_f[c, :] = (float(x), float(y), sf)

    ## REMOVE SIMILAR FEATURES (KEEP BIGGER ONE)

    i = 0
    while i < num_candidates - 1:
        c_i = candidates_f[i, :]
        j = i + 1
        while j < num_candidates:
            c_j = candidates_f[j, :]
            if abs(c_i[2] - c_j[2]) <= step and abs(c_i[0] - c_j[0]) <= 3 and abs(c_i[1] - c_j[1]) <= 3:
                if c_i[2] > c_j[2]:
                    candidates_f[j, :] = candidates_f[num_candidates - 1, :]
                else:
                    candidates_f[i, :] = candidates_f[num_candidates - 1, :]
                num_candidates -= 1
            j += 1
        i += 1

    return candidates_f[:num_candidates, :]


@jit(float64[:, :,:](float64[:, :, :], float64, float64), nopython=True, cache=True)
def edge_pixels(image, sigma, min_magnitude):
    h,w,d = image.shape

    ## COMPUTE AMPLITUDES AND GRADIENT DIRECTIONS

    filter_mat_x = convolution.derivative_of_gaussian(sigma, 0)
    filter_mat_y = convolution.derivative_of_gaussian(sigma, 1)

    Gx = convolution.fast_sw_convolution(image, filter_mat_x, convolution.SAME)[:,:,0]
    Gy = convolution.fast_sw_convolution(image, filter_mat_y, convolution.SAME)[:,:,0]

    magnitudes = np.sqrt(np.square(Gx) + np.square(Gy))
    gradient_directions = np.arctan2(Gy, Gx) * 180 / np.pi

    ## CREATE EDGE MAP

    edge_map = np.zeros((h,w), dtype=np.float64)

    for x in range(1, h - 1):
        for y in range(1, w - 1):
            magnitude = magnitudes[x, y]
            if magnitude < min_magnitude:
                continue
            direction = gradient_directions[x, y]
            if direction < 0:
                direction += 360
            if direction > 180:
                direction -= 180

            if direction < 22.5 or direction >= 157.5:
                if magnitudes[x + 1, y] <= magnitude and magnitudes[x - 1, y] <= magnitude:
                    edge_map[x, y] = 255
            elif direction >= 22.5 and direction < 67.5:
                if magnitudes[x - 1, y - 1] <= magnitude and magnitudes[x + 1, y + 1] <= magnitude:
                    edge_map[x, y] = 255
            elif direction >= 67.5 and direction < 112.5:
                if magnitudes[x, y - 1] <= magnitude and magnitudes[x, y + 1] <= magnitude:
                    edge_map[x, y] = 255
            elif direction >= 112.5 and direction < 157.5:
                if magnitudes[x + 1, y - 1] <= magnitude and magnitudes[x - 1, y + 1] <= magnitude:
                    edge_map[x, y] = 255

    return edge_map.reshape((h,w,1))


@jit(types.Tuple((float64[:, :], float64[:, :, :]))(float64[:, :, :], float64, float64, float64, int64), nopython=True,
     cache=True)
def hough_lines(image, sigma, t_amp, n_best, show_hough_space):
    h,w,d = image.shape
    filter_mat_x = convolution.derivative_of_gaussian(sigma, 0)
    filter_mat_y = convolution.derivative_of_gaussian(sigma, 1)

    Gx = convolution.fast_sw_convolution(image, filter_mat_x, convolution.SAME)[:,:,0]
    Gy = convolution.fast_sw_convolution(image, filter_mat_y, convolution.SAME)[:,:,0]

    amplitudes = np.sqrt(np.square(Gx) + np.square(Gy))

    image_diag = int(np.sqrt(h*h + w*w))
    sampling_size_x = image_diag
    sampling_size_y = 180  # 1 degree steps

    hough_space = np.zeros((sampling_size_x, sampling_size_y), dtype=np.float64)

    for x in range(h):
        for y in range(w):
            v_ = np.sqrt(x * x + y * y)
            amp = amplitudes[x, y]
            if amp < t_amp:
                continue
            phi = 0.0
            if y > 0:
                phi = math.atan(x / y)  # from x-axes ccw
            for s in range(sampling_size_y):
                omega = s * 180 / sampling_size_y
                theta = omega * np.pi / 180.0 - phi
                r = math.cos(theta) * v_
                hough_space[int(r * (sampling_size_x / 2) / image_diag + sampling_size_x / 2), s] += amp

    hough_space = hough_space.reshape((sampling_size_x, sampling_size_y, 1))

    filter_blur = convolution.gauss_filter(0.4)
    hough_space = convolution.fast_sw_convolution(hough_space, filter_blur, convolution.SAME)

    candidates = tools.non_max_suppression_3d(hough_space, 10)
    candidates = tools.get_n_best_3d(candidates, hough_space, n_best)
    candidates = candidates.astype(np.float64)

    if show_hough_space:
        r = min(sampling_size_x, sampling_size_y) / 25  # radius = 1/25 * shorter window side
        for c in range(candidates.shape[0]):
            candidate = candidates[c, :]
            candidate[2] = r
        hough_space = tools.convert_to_3channel(tools.normalize(hough_space))
        tools.draw_circles(hough_space, candidates)

    for c in range(candidates.shape[0]):
        candidate = candidates[c, :]
        candidate[0] -= sampling_size_x / 2
        candidate[0] *= image_diag / (sampling_size_x / 2)
        candidate[1] *= np.pi / sampling_size_y

    return (candidates, hough_space)


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def hough_lines_nice(image, sigma):
    filter_mat_x = convolution.derivative_of_gaussian(sigma, 0)
    filter_mat_y = convolution.derivative_of_gaussian(sigma, 1)

    Gx = convolution.fast_sw_convolution(image, filter_mat_x, convolution.SAME)
    Gy = convolution.fast_sw_convolution(image, filter_mat_y, convolution.SAME)
    amplitude = np.sqrt(np.square(Gx) + np.square(Gy))

    image_diag = int((image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5)
    sampling_size_x = image.shape[0]
    sampling_size_y = image.shape[1]

    hough_space = np.zeros((sampling_size_x, sampling_size_y, image.shape[2]), dtype=np.float64)

    height = amplitude.shape[0]
    width = amplitude.shape[1]

    for x in range(height):
        for y in range(width):
            v_ = (x ** 2 + y ** 2) ** 0.5
            amp = amplitude[x, y, :]

            phi = 0.0
            if y > 0:
                phi = math.atan(x / y)  # from x-axes ccw
            for s in range(sampling_size_y):
                omega = 360.0 * s / sampling_size_y
                theta = omega * np.pi / 180.0 - phi
                r = math.cos(theta) * v_
                hough_space[int(r * (sampling_size_x / 2) / image_diag + sampling_size_x / 2), s, :] += amp

    # hough_space = hough_space.reshape((sampling_size_x, sampling_size_y, 1))
    filter_blur = convolution.gauss_filter(1)
    hough_space = convolution.fast_sw_convolution(hough_space, filter_blur, convolution.SAME)
    return hough_space
