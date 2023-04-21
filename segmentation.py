import numpy as np
import tools
from numba import jit, float64, int64, boolean, types
import convolution

GROUPNAME = 'Segmentation'

WATERSHED = 'Watershed Transform'
GRAY_THRESHOLDING = 'Gray Value Thresholding'
RGB_THRESHOLDING = 'RGB Thresholding'
HSV_THRESHOLDING = 'HSV Thresholding'
REGION_GROWING = 'Region Growing'
MEAN_SHIFT = 'Mean Shift'
SLIC = 'Slic'
FILTER_BANK_CLUSTERING = 'Filter Bank Clustering'
CONNECTED_COMPONENTS = 'Connected Components'

OPS = (WATERSHED, GRAY_THRESHOLDING, RGB_THRESHOLDING, HSV_THRESHOLDING, REGION_GROWING, MEAN_SHIFT, SLIC,
       CONNECTED_COMPONENTS, FILTER_BANK_CLUSTERING)


def apply(image, operation, p1, p2, p3, p4):
    gray_image = tools.convert_to_1channel(image)

    if operation == WATERSHED:
        sigma = float(p1)
        seed_threshold = float(p3)
        shed_only = (p2 == '1')

        label_map = watershed_transform(gray_image, sigma, seed_threshold)

        if shed_only:
            return (label_map == 0).reshape((label_map.shape[0], label_map.shape[1], 1)).astype(np.int64) * 255
        else:
            return tools.label_map2label_image(label_map)

    elif operation == GRAY_THRESHOLDING:
        g_min = float(p1)
        g_max = float(p2)
        mask = gray_value_thresholding(gray_image, g_min, g_max)
        return tools.binary2gray(mask)

    elif operation == RGB_THRESHOLDING:
        r = p1.split(',')
        r_min = float(r[0])
        r_max = float(r[1])

        g = p2.split(',')
        g_min = float(g[0])
        g_max = float(g[1])

        b = p3.split(',')
        b_min = float(b[0])
        b_max = float(b[1])

        mask = rgb_thresholding(image, r_min, r_max, g_min, g_max, b_min, b_max)
        return tools.binary2gray(mask)

    elif operation == HSV_THRESHOLDING:
        h = p1.split(',')
        h_min = float(h[0])
        h_max = float(h[1])

        s = p2.split(',')
        s_min = float(s[0])
        s_max = float(s[1])

        v = p3.split(',')
        v_min = float(v[0])
        v_max = float(v[1])

        mask = hsv_thresholding(image, h_min, h_max, s_min, s_max, v_min, v_max)
        return tools.binary2gray(mask)

    elif operation == REGION_GROWING:
        h_g = float(p1)
        if image.shape[2] == 1:

            label_map = region_growing_gray(image, h_g)
        else:
            # label_map = region_growing_color(image, h_g)
            label_map = region_growing_lab(image, h_g)

        if p4 == 'average' or p4 == 'a':
            label_image = tools.label_map2label_image_avg(image, label_map)
            return label_image
        else:
            label_image = tools.label_map2label_image(label_map)
            return label_image

    elif operation == MEAN_SHIFT:
        h_g = float(p1)
        use_spacial = (p2 == '1')
        h_s = float(p3)
        if image.shape[2] == 1:
            if use_spacial:
                return mean_shift_gray_space(image, h_g, h_s)
            else:
                return mean_shift_gray(image, h_g)
        else:
            if use_spacial:
                return mean_shift_color_space(image, h_g, h_s)
            else:
                return mean_shift_color(image, h_g)

    elif operation == SLIC:
        num_pixels = int(p1)
        compactness = float(p2)
        E = float(p3)
        label_map = slic(image, num_pixels, compactness, E)

        if p4 == 'average' or p4 == 'a':
            label_image = tools.label_map2label_image_avg(image, label_map)
            return label_image
        else:
            label_image = tools.label_map2label_image(label_map)
            return label_image

    elif operation == CONNECTED_COMPONENTS:
        label_map = connected_components(image)
        label_image = tools.label_map2label_image(label_map)
        return label_image

    elif operation == FILTER_BANK_CLUSTERING:
        sigma_min = float(p1)
        sigma_max = float(p2)
        k = int(p3)
        th = float(p4)
        label_map = filter_bank_clustering(image,sigma_min, sigma_max, k, th)
        return tools.label_map2label_image(label_map)


@jit(int64[:, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def watershed_transform(image, sigma, seed_threshold):  # Meyer's flooding algorithm
    h, w, _ = image.shape

    filter_dgx = convolution.derivative_of_gaussian(sigma, 0)
    filter_dgy = convolution.derivative_of_gaussian(sigma, 1)
    dgx = convolution.fast_sw_convolution(image, filter_dgx, convolution.SAME)
    dgy = convolution.fast_sw_convolution(image, filter_dgy, convolution.SAME)
    amplitude_map = (np.sqrt(dgx * dgx + dgy * dgy)).reshape(h, w)

    max_amp_plus = np.max(amplitude_map) + 1

    (gmx, gmy), gmv = tools.get_min_coords_2d_threshold(amplitude_map, max_amp_plus)
    num_candidates = 1

    candidate_map = np.ones((h, w), dtype=np.float64) * max_amp_plus
    visited_map = np.zeros((h, w), dtype=np.bool_)
    label_map = np.zeros((h, w), dtype=np.int64)

    next_label = 1

    candidate_map[gmx, gmy] = gmv
    label_map[gmx, gmy] = next_label
    amplitude_map[gmx, gmy] = max_amp_plus

    # second best minimum
    (gmx, gmy), gmv = tools.get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

    while num_candidates > 0:
        (cx, cy), cv = tools.get_min_coords_2d_threshold(candidate_map, max_amp_plus)

        # check for new seeds
        if cv - seed_threshold > gmv:
            if not (candidate_map[gmx, gmy] < max_amp_plus or visited_map[gmx, gmy]):
                candidate_map[gmx, gmy] = gmv
                num_candidates += 1
                next_label += 1
                label_map[gmx, gmy] = next_label
                cx = gmx
                cy = gmy

            (gmx, gmy), gmv = tools.get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

        # remove candidate from candidates and add to visited
        amplitude_map[cx, cy] = max_amp_plus
        candidate_map[cx, cy] = max_amp_plus
        visited_map[cx, cy] = True
        num_candidates -= 1

        neighbours = tools.get_valid_neighbours(h, w, cx, cy)
        num_neighbours = neighbours.shape[0]

        can_be_labeled = True
        label_vote = 0
        for n in range(num_neighbours):
            nx, ny = neighbours[n, :]
            if not (candidate_map[nx, ny] < max_amp_plus or visited_map[nx, ny]):
                candidate_map[nx, ny] = amplitude_map[nx, ny]
                num_candidates += 1

            label = label_map[nx, ny]

            if label == 0:
                continue

            if label_vote == 0:
                label_vote = label
            elif not label_vote == label:
                can_be_labeled = False

        if can_be_labeled and (not label_map[cx, cy]):
            label_map[cx, cy] = label_vote

    return label_map


### ========== THRESHOLDING =================

@jit(boolean[:, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def gray_value_thresholding(image, g_min, g_max):
    mask = np.logical_and(image[:, :, 0] >= g_min, image[:, :, 0] <= g_max)
    return mask


@jit(boolean[:, :](float64[:, :, :], float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def rgb_thresholding(image, r_min, r_max, g_min, g_max, b_min, b_max):
    h, w, _ = image.shape
    red_mask = np.logical_and(image[:, :, 0] >= r_min, image[:, :, 0] <= r_max)
    green_mask = np.logical_and(image[:, :, 1] >= g_min, image[:, :, 1] <= g_max)
    blue_mask = np.logical_and(image[:, :, 2] >= b_min, image[:, :, 2] <= b_max)
    mask = np.logical_and(np.logical_and(red_mask, green_mask), blue_mask)
    return mask


@jit(boolean[:, :](float64[:, :, :], float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def hsv_thresholding(hsv_image, h_min, h_max, s_min, s_max, v_min, v_max):
    hsv_image = tools.rgb2hsv(hsv_image)

    if h_min > h_max:
        hue_mask = np.logical_or(hsv_image[:, :, 0] >= h_min, hsv_image[:, :, 0] <= h_max)
    else:
        hue_mask = np.logical_and(hsv_image[:, :, 0] >= h_min, hsv_image[:, :, 0] <= h_max)
    sat_mask = np.logical_and(hsv_image[:, :, 1] >= s_min, hsv_image[:, :, 1] <= s_max)
    val_mask = np.logical_and(hsv_image[:, :, 2] >= v_min, hsv_image[:, :, 2] <= v_max)
    mask = np.logical_and(np.logical_and(hue_mask, sat_mask), val_mask)
    return mask


# ========== REGION GROWING ==============

@jit(int64[:, :](float64[:, :, :], float64), nopython=True, cache=True)
def region_growing_gray(image, h_g):
    h, w, _ = image.shape

    num_unlabeled = h * w
    num_labels = 0
    label_means = np.zeros((h * w), dtype=np.float64)
    label_sizes = np.zeros((h * w), dtype=np.int64)
    label_map = np.ones((h, w), dtype=np.int64) * -1  # -1 means unlabeled
    candidate_list = np.zeros((h * w, 2), dtype=np.int64)  # (index,coordinates)

    labeled_rows = 0

    while num_unlabeled > 0:
        ## SEARCH FOR UNLABELED PIXEL AS SEED

        found_seed = False
        for sx in range(labeled_rows, h):
            for sy in range(w):
                if label_map[sx, sy] == -1:
                    found_seed = True
                    break
            if found_seed:
                break
            labeled_rows += 1

        new_label = num_labels
        num_labels += 1

        label_map[sx, sy] = new_label
        label_means[new_label] = image[sx, sy, 0]
        label_sizes[new_label] = 1

        candidate_list[0, :] = (sx, sy)
        num_candidates = 1
        num_unlabeled -= 1

        while num_candidates > 0:
            num_candidates -= 1
            cx, cy = candidate_list[num_candidates, :]  # pop last candidate
            cg = image[cx, cy, 0]

            valid_neighbours = tools.get_valid_neighbours(h, w, cx, cy)
            num_neighbours = valid_neighbours.shape[0]

            for i in range(num_neighbours):
                nx, ny = valid_neighbours[i, :]
                nl = label_map[nx, ny]
                if nl >= 0:
                    continue  # if neighbour is already labeled

                if abs(label_means[new_label] - cg) <= h_g:
                    label_map[nx, ny] = new_label
                    old_size = label_sizes[new_label]
                    new_size = old_size + 1
                    label_means[new_label] = (image[nx, ny, 0] + label_means[new_label] * old_size) / new_size
                    label_sizes[new_label] = new_size

                    candidate_list[num_candidates, :] = (nx, ny)
                    num_candidates += 1
                    num_unlabeled -= 1

    return label_map


@jit(int64[:, :](float64[:, :, :], float64), nopython=True, cache=True)
def region_growing_color(image, h_g):
    h, w, _ = image.shape

    num_unlabeled = h * w
    num_labels = 0
    label_means = np.zeros((h * w, 3), dtype=np.float64)
    label_sizes = np.zeros((h * w), dtype=np.int64)
    label_map = np.ones((h, w), dtype=np.int64) * -1  # -1 means unlabeled
    candidate_list = np.zeros((h * w, 2), dtype=np.int64)  # (index,coordinates)

    labeled_rows = 0

    while num_unlabeled > 0:
        ## SEARCH FOR UNLABELED PIXEL AS SEED

        found_seed = False
        for sx in range(labeled_rows, h):
            for sy in range(w):
                if label_map[sx, sy] == -1:
                    found_seed = True
                    break
            if found_seed:
                break
            labeled_rows += 1

        new_label = num_labels
        num_labels += 1

        label_map[sx, sy] = new_label
        label_means[new_label] = image[sx, sy, :]
        label_sizes[new_label] = 1

        candidate_list[0, :] = (sx, sy)
        num_candidates = 1
        num_unlabeled -= 1

        while num_candidates > 0:
            num_candidates -= 1
            cx, cy = candidate_list[num_candidates, :]  # pop last candidate
            cg = image[cx, cy, :]

            valid_neighbours = tools.get_valid_neighbours(h, w, cx, cy)
            num_neighbours = valid_neighbours.shape[0]

            for i in range(num_neighbours):
                nx, ny = valid_neighbours[i, :]
                nl = label_map[nx, ny]
                if nl >= 0:
                    continue  # if neighbour is already labeled

                if np.linalg.norm(label_means[new_label] - cg) <= h_g:
                    label_map[nx, ny] = new_label
                    old_size = label_sizes[new_label]
                    new_size = old_size + 1
                    label_means[new_label] = (image[nx, ny, :] + label_means[new_label] * old_size) / new_size
                    label_sizes[new_label] = new_size

                    candidate_list[num_candidates, :] = (nx, ny)
                    num_candidates += 1
                    num_unlabeled -= 1

    return label_map


@jit(int64[:, :](float64[:, :, :], float64), nopython=True, cache=True)
def region_growing_lab(image, h_g):
    h, w, _ = image.shape
    Lab_image = tools.rgb2lab(image)

    num_unlabeled = h * w
    num_labels = 0
    label_means = np.zeros((h * w, 3), dtype=np.float64)
    label_sizes = np.zeros((h * w), dtype=np.int64)
    label_map = np.ones((h, w), dtype=np.int64) * -1  # -1 means unlabeled
    candidate_list = np.zeros((h * w, 2), dtype=np.int64)  # (index,coordinates)

    labeled_rows = 0

    while num_unlabeled > 0:
        ## SEARCH FOR UNLABELED PIXEL AS SEED

        found_seed = False
        for sx in range(labeled_rows, h):
            for sy in range(w):
                if label_map[sx, sy] == -1:
                    found_seed = True
                    break
            if found_seed:
                break
            labeled_rows += 1

        new_label = num_labels
        num_labels += 1

        label_map[sx, sy] = new_label
        label_means[new_label] = Lab_image[sx, sy, :]
        label_sizes[new_label] = 1

        candidate_list[0, :] = (sx, sy)
        num_candidates = 1
        num_unlabeled -= 1

        while num_candidates > 0:
            num_candidates -= 1
            cx, cy = candidate_list[num_candidates, :]  # pop last candidate
            cg = Lab_image[cx, cy, :]

            valid_neighbours = tools.get_valid_neighbours(h, w, cx, cy)
            num_neighbours = valid_neighbours.shape[0]

            for i in range(num_neighbours):
                nx, ny = valid_neighbours[i, :]
                nl = label_map[nx, ny]
                if nl >= 0:
                    continue  # if neighbour is already labeled

                if np.linalg.norm(label_means[new_label] - cg) <= h_g:
                    label_map[nx, ny] = new_label
                    old_size = label_sizes[new_label]
                    new_size = old_size + 1
                    label_means[new_label] = (Lab_image[nx, ny, :] + label_means[new_label] * old_size) / new_size
                    label_sizes[new_label] = new_size

                    candidate_list[num_candidates, :] = (nx, ny)
                    num_candidates += 1
                    num_unlabeled -= 1

    return label_map

# FROM LAB

@jit(nopython=True, cache=True)
def region_growing_lab_gray(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on a gray-value image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        max_dist : float
            maximum gray-value distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    h, w, d = I.shape
    assert d == 1, "Only single channel images supported!"
    G = I[:, :, 0]
    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_gray = G[sx, sy]
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_gray = sum_gray  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ng = G[nx, ny]
                    nc = S[nx, ny]
                    g_dist = abs(ng - mean_gray)
                    if nc != current_seg_id and nc >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = nc

                    if nc < 0 and g_dist < max_dist:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1
                        sum_gray += ng
                        mean_gray = sum_gray / num_pixels
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    ### SMOOTHING
    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float64)
        hfs = n // 2
        S_ext = __extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = connected_components(R)

    return S


@jit(nopython=True, cache=True)
def region_growing_lab_rgb(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on a RGB image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        max_dist : float
            maximum euclidean rgb distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    max_dist_sq = max_dist ** 2
    h, w, d = I.shape
    assert d == 3, "Only color images supported!"

    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(I[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = I[nx, ny]
                    ns = S[nx, ny]
                    g_dist = np.sum(np.square(ncol - mean_colors))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns

                    if ns < 0 and g_dist < max_dist_sq:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1
                        sum_colors += ncol
                        mean_colors = sum_colors / num_pixels
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    ### SMOOTHING
    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float64)
        hfs = n // 2
        S_ext = __extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = get_connected_components(R)

    return S


@jit(nopython=True, cache=True)
def region_growing_lab_hsv(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on an color image in HSV space.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image
        max_dist : float
            maximum hsv distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        The hsv distance is a experimental metric which weights the hue distance,
        depending on the mean saturation and value of a pixel and region mean.
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    max_dist_sq = max_dist ** 2
    h, w, d = I.shape
    assert d == 3, "Only color images supported!"
    HSV = rgb2hsv(I)

    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(HSV[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = HSV[nx, ny]
                    ns = S[nx, ny]

                    dist = ncol - mean_colors
                    if dist[0] > 180:
                        dist[0] -= 360
                    elif dist[0] < -180:
                        dist[0] += 360

                    dist[0] *= (ncol[1] + mean_colors[1] + ncol[2] + mean_colors[2]) / (4 * 180)

                    g_dist = np.sum(np.square(dist))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns

                    if ns < 0 and g_dist < max_dist_sq:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1

                        sum_colors += ncol
                        mean_colors = sum_colors / num_pixels
                        if mean_colors[0] < 0:
                            mean_colors[0] += 360
                        elif mean_colors[0] > 360:
                            mean_colors[0] -= 360
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    # ====== SMOOTHING =========

    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float64)
        hfs = n // 2
        S_ext = __extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = get_connected_components(R)

    return S


# =========== MEAN SHIFT =================

@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def mean_shift_gray(image, h_g):
    h, w, _ = image.shape
    f = -1.0 / (2.0 * h_g * h_g)

    num_modes = h * w
    modes = np.ones((num_modes, 2), dtype=np.float64)  # gray value, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value = image[x, y, 0]
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value, m_old_count = modes[m_old, :]
                if abs(m_new_value - m_old_value) < h_g:
                    ## MERGE
                    sum_count = (m_old_count + m_new_count)
                    modes[m_old, 0] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                    modes[m_old, 1] = sum_count

                    exists = True
                    break

            if not exists:
                modes[m_new, 0] = m_new_value
                modes[m_new, 1] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        mode_changed = False
        print(i, num_modes)

        ## MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value, m1_count = modes[m1_index, :]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value, m2_count = modes[m2_index, :]
                if abs(m1_value - m2_value) > h_g:
                    m2_index += 1
                    continue

                # MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, 0] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 1] = sum_count

                # REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                mode_changed = True

            m1_index += 1

        ## GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, 0]
            nomin = 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value, m2_count = modes[m2_index, :]
                dist = abs(m1_value - m2_value)
                if dist > 3 * h_g:
                    continue
                weight = np.exp(f * dist * dist) * m2_count
                nomin += m2_value * weight
                denom += weight

            new_value = nomin / denom
            if new_value != m1_value:
                mode_changed = True
                modes[m1_index, 0] = new_value

        if not mode_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            gv = image[x, y, 0]
            nearest_dist = 1000000.0
            nearest_mode = 0
            for m in range(num_modes):
                mv = modes[m, 0]
                dist = abs(gv - mv)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_mode = m
            image[x, y, 0] = modes[nearest_mode, 0]

    return image


@jit(float64[:, :, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def mean_shift_gray_space(image, h_g, h_s):
    nominator = np.zeros((3), dtype=np.float64)
    p_value = np.zeros((3), dtype=np.float64)
    h, w, _ = image.shape
    f_g = -1.0 / (2.0 * h_g * h_g)
    f_s = -1.0 / (2.0 * h_s * h_s)

    h_s_sq = h_s * h_s

    num_modes = h * w
    modes = np.ones((num_modes, 4), dtype=np.float64)  # gray value,x,y, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((3), dtype=np.float64)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[0] = image[x, y, 0]
            m_new_value[1] = x
            m_new_value[2] = y
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :3]
                m_old_count = modes[m_old, 3]
                d_g = m_new_value[0] - m_old_value[0]

                if abs(d_g) < h_g:
                    d_xy = m_new_value[1:3] - m_old_value[1:3]
                    d_s_sq = np.sum(d_xy * d_xy)
                    d_s = np.sqrt(d_s_sq)
                    if d_s < h_s:
                        ## MERGE
                        sum_count = (m_old_count + m_new_count)
                        modes[m_old, :3] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                        modes[m_old, 3] = sum_count

                        exists = True
                        break

            if not exists:
                modes[m_new, :3] = m_new_value
                modes[m_new, 3] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :3]
            m1_count = modes[m1_index, 3]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]
                d_g = m1_value[0] - m2_value[0]
                if abs(d_g) > h_g:
                    m2_index += 1
                    continue

                d_xy = m2_value[1:3] - m1_value[1:3]
                d_s_sq = np.sum(d_xy * d_xy)
                if d_s_sq > h_s_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :3] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 3] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :3]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]
                d_g = m1_value[0] - m2_value[0]
                if abs(d_g) > 3 * h_g:
                    continue
                d_xy = m2_value[1:3] - m1_value[1:3]
                d_s_sq = np.sum(d_xy * d_xy)
                d_s = np.sqrt(d_s_sq)
                if d_s > 3 * h_s:
                    continue
                ## UPDATE
                weight = np.exp(f_g * d_g * d_g) * np.exp(f_s * d_s_sq) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2]:
                modes_changed = True
                modes[m1_index, :3] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = (image[x, y, 0], float(x), float(y))
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                dg, dx, dy = p_value[:] - modes[m, :3]
                dist_s_sq = dx ** 2 + dy ** 2
                dist_g_sq = dg * dg
                weight_c = np.exp(f_g * dist_g_sq)
                weight_s = np.exp(f_s * dist_s_sq)
                weight = weight_c * weight_s
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, 0] = modes[best_mode, 0]

    return image


@jit(float64[:, :, :](float64[:, :, :], float64), nopython=True, cache=True)
def mean_shift_color(image, h_c):
    nominator = np.zeros((3), dtype=np.float64)
    p_value = np.zeros((3), dtype=np.float64)
    h, w, _ = image.shape
    f_c = -1.0 / (2.0 * h_c * h_c)

    h_c_sq = h_c * h_c

    num_modes = h * w
    modes = np.ones((num_modes, 4), dtype=np.float64)  # r,g,b, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((3), dtype=np.float64)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[:] = image[x, y, :]
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :3]
                m_old_count = modes[m_old, 3]

                d_rgb = m_new_value[1:3] - m_old_value[1:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_rgb_sq)
                if d_c < h_c:
                    ## MERGE
                    sum_count = (m_old_count + m_new_count)
                    modes[m_old, :3] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                    modes[m_old, 3] = sum_count

                    exists = True
                    break

            if not exists:
                modes[m_new, :3] = m_new_value
                modes[m_new, 3] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :3]
            m1_count = modes[m1_index, 3]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                if d_rgb_sq > h_c_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :3] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 3] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :3]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_rgb_sq)
                if d_c > 3 * h_c:
                    continue
                ## UPDATE
                weight = np.exp(f_c * d_c * d_c) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2]:
                modes_changed = True
                modes[m1_index, :3] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = image[x, y, :]
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                d_rgb = p_value[:] - modes[m, :3]
                dist_c_sq = np.sum(d_rgb * d_rgb)
                weight = np.exp(f_c * dist_c_sq)
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, :] = modes[best_mode, :3]

    return image


@jit(float64[:, :, :](float64[:, :, :], float64, float64), nopython=True, cache=True)
def mean_shift_color_space(image, h_c, h_s):
    nominator = np.zeros((5), dtype=np.float64)
    p_value = np.zeros((5), dtype=np.float64)
    h, w, _ = image.shape
    f_c = -1.0 / (2.0 * h_c * h_c)
    f_s = -1.0 / (2.0 * h_s * h_s)

    h_c_sq = h_c * h_c
    h_s_sq = h_s * h_s

    num_modes = h * w
    modes = np.ones((num_modes, 6), dtype=np.float64)  # gray value,x,y, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((5), dtype=np.float64)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[0] = image[x, y, 0]
            m_new_value[1] = image[x, y, 1]
            m_new_value[2] = image[x, y, 2]
            m_new_value[3] = x
            m_new_value[4] = y
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :5]
                m_old_count = modes[m_old, 5]

                d_rgb = m_new_value[:3] - m_old_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)

                if d_c_sq < h_s_sq:
                    d_xy = m_new_value[3:5] - m_old_value[3:5]
                    d_s_sq = np.sum(d_xy * d_xy)
                    if d_s_sq < h_s_sq:
                        ## MERGE
                        sum_count = (m_old_count + m_new_count)
                        modes[m_old, :5] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                        modes[m_old, 5] = sum_count

                        exists = True
                        break

            if not exists:
                modes[m_new, :5] = m_new_value
                modes[m_new, 5] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :5]
            m1_count = modes[m1_index, 5]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :5]
                m2_count = modes[m2_index, 5]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)
                if d_c_sq > h_c_sq:
                    m2_index += 1
                    continue

                d_xy = m2_value[3:5] - m1_value[3:5]
                d_s_sq = np.sum(d_xy * d_xy)
                if d_s_sq > h_s_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :5] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 5] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :5]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :5]
                m2_count = modes[m2_index, 5]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_c_sq)
                if d_c > 3 * h_c:
                    continue

                d_xy = m2_value[3:5] - m1_value[3:5]
                d_s_sq = np.sum(d_xy * d_xy)
                d_s = np.sqrt(d_s_sq)
                if d_s > 3 * h_s:
                    continue

                ## UPDATE
                weight = np.exp(f_c * d_c_sq) * np.exp(f_s * d_s_sq) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2] \
                    or new_value[3] != m1_value[3] or new_value[4] != m1_value[4]:
                modes_changed = True
                modes[m1_index, :5] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = (image[x, y, 0], image[x, y, 1], image[x, y, 2], float(x), float(y))
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                dr, dg, db, dx, dy = p_value[:] - modes[m, :5]
                dist_c_sq = dr * dr + dg * dg + db * db
                dist_s_sq = dx * dx + dy * dy
                weight_c = np.exp(f_c * dist_c_sq)
                weight_s = np.exp(f_s * dist_s_sq)
                weight = weight_c * weight_s
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, :] = modes[best_mode, :3]

    return image


# ============= SUPER PIXELS ==================

@jit(int64[:, :](float64[:, :, :], int64, float64, float64), nopython=True, cache=True)
def slic(image, K, m, E):
    # K = number of superpixels
    # m = compactness

    h, w, d = image.shape
    N = h * w
    S = (N / K) ** 0.5

    lab_image = tools.rgb2lab(image)
    label_map = np.zeros((h, w), dtype=np.int64)

    num_clusters = int((h // S + 1) * (w // S + 1))
    print(num_clusters)
    # S = (N / num_clusters) ** 0.5
    # print(S)

    c = 0

    avgs = np.zeros((num_clusters, 5), dtype=np.float64)
    num_pixels = np.zeros((num_clusters), dtype=np.int64)
    clusters = np.zeros((num_clusters, 5), dtype=np.float64)  # x,y,l,a,b
    xc = 0
    while xc <= h:
        yc = 0
        while yc <= w:
            L, a, b = lab_image[int(round(xc)), int(round(yc)), :]
            clusters[c, :] = (xc, yc, L, a, b)
            c += 1
            yc += S
        xc += S

    i = 0
    while True:
        i += 1
        avgs *= 0
        num_pixels *= 0
        for x in range(h):
            for y in range(w):
                nearest_cluster = 0
                nearest_distance = 9999999
                L, a, b = lab_image[x, y, :]
                for c in range(num_clusters):
                    xc, yc, Lc, ac, bc = clusters[c, :]
                    dx = x - xc
                    dy = y - yc
                    if abs(dx) > S or abs(dy) > S:
                        continue

                    dL = L - Lc
                    da = a - ac
                    db = b - bc

                    Dxy = (dx * dx + dy * dy) ** 0.5
                    # Dxy = abs(dx) + abs(dy)
                    Dlab = (dL * dL + da * da + db * db) ** 0.5
                    Ds = Dlab + m / S * Dxy

                    if Ds < nearest_distance:
                        nearest_distance = Ds
                        nearest_cluster = c

                avgs[nearest_cluster, 0] += x
                avgs[nearest_cluster, 1] += y
                avgs[nearest_cluster, 2] += L
                avgs[nearest_cluster, 3] += a
                avgs[nearest_cluster, 4] += b

                num_pixels[nearest_cluster] += 1
                label_map[x, y] = nearest_cluster

        clusters_pre = np.copy(clusters)
        for c in range(num_clusters):
            clusters[c] = avgs[c, :] / num_pixels[c]
        max_diff = np.max(np.sum(np.abs(clusters - clusters_pre), 1))

        if max_diff < E:
            break

    return label_map


# ============= POST PROCESSING ===============

@jit(int64[:, :](float64[:, :, :]), nopython=True, cache=True)
def connected_components(image):
    h, w, d = image.shape

    ### MAP COORDINATES TO HORIZONTAL LABEL
    label_map_horizontal = np.zeros((h, w), dtype=np.int64)
    next_label = 1
    for x in range(h):
        for y in range(w):
            merge_h = y > 0
            if merge_h:
                for z in range(d):
                    if image[x, y - 1, z] != image[x, y, z]:
                        merge_h = False
                        break

            if merge_h:
                label_map_horizontal[x, y] = label_map_horizontal[x, y - 1]

            else:
                merge_v = x > 0
                if merge_v:
                    for z in range(d):
                        if image[x - 1, y, z] != image[x, y, z]:
                            merge_v = False
                            break

                if merge_v:
                    label_map_horizontal[x, y] = label_map_horizontal[x - 1, y]

                else:
                    label_map_horizontal[x, y] = next_label
                    next_label += 1

    ### MAP HORIZONTAL INDICES TO FINAL INDICES
    final_labels_length = next_label
    final_labels = np.ones((final_labels_length), dtype=np.int64) * -1
    next_final_label = 0
    for x in range(h):
        for y in range(w):
            try_merge = x > 0
            h_label = label_map_horizontal[x, y]

            merge = try_merge
            if try_merge:
                upper_h_label = label_map_horizontal[x - 1, y]
                if h_label == upper_h_label or final_labels[h_label] == final_labels[upper_h_label]:
                    merge = False
                else:
                    for z in range(d):
                        if image[x - 1, y, z] != image[x, y, z]:
                            merge = False
                            break
                if merge:
                    if final_labels[h_label] == -1:
                        final_labels[h_label] = final_labels[upper_h_label]
                    elif final_labels[upper_h_label] != final_labels[h_label]:
                        to_change_final = final_labels[h_label]
                        for i in range(final_labels_length):
                            if final_labels[i] == to_change_final:
                                final_labels[i] = final_labels[upper_h_label]

            if not merge and final_labels[h_label] == -1:
                final_labels[h_label] = next_final_label
                next_final_label += 1

    ### MAP FINAL INDICES TO INDICES WITHOUT GAPS (STARTING FROM ZERO)
    no_gap_labels_length = next_final_label
    no_gap_labels = np.ones((no_gap_labels_length), dtype=np.int64) * -1
    no_gap_index = 0
    for x in range(h):
        for y in range(w):
            if no_gap_labels[final_labels[label_map_horizontal[x, y]]] == -1:
                no_gap_labels[final_labels[label_map_horizontal[x, y]]] = no_gap_index
                no_gap_index += 1

    label_map = np.zeros((h, w), dtype=np.int64)
    for x in range(h):
        for y in range(w):
            label_map[x, y] = no_gap_labels[final_labels[label_map_horizontal[x, y]]]

    return label_map

# only 2 neighbours
def get_border_pixels(C):
    h, w = C.shape
    num_classes = np.max(C)
    B = np.zeros((h, w), dtype=np.int64)

    for c in range(1, num_classes):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if C[x, y] == c:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF CLASS 'c'
        search_dir = 0
        current_pixel = start_pixel
        while True:
            cpx, cpy = current_pixel
            B[cpx, cpy] = c
            while True:
                if search_dir == 0 and cpx > 0 and C[cpx - 1, cpy] == c:
                    current_pixel = (cpx - 1, cpy)
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and C[cpx - 1, cpy + 1] == c:
                    current_pixel = (cpx - 1, cpy + 1)
                    break
                elif search_dir == 90 and cpy < w - 1 and C[cpx, cpy + 1] == c:
                    current_pixel = (cpx, cpy + 1)
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and C[cpx + 1, cpy + 1] == c:
                    current_pixel = (cpx + 1, cpy + 1)
                    break
                elif search_dir == 180 and cpx < h - 1 and C[cpx + 1, cpy] == c:
                    current_pixel = (cpx + 1, cpy)
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and C[cpx + 1, cpy - 1] == c:
                    current_pixel = (cpx + 1, cpy - 1)
                    break
                elif search_dir == 270 and cpy > 0 and C[cpx, cpy - 1] == c:
                    current_pixel = (cpx, cpy - 1)
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and C[cpx - 1, cpy - 1] == c:
                    current_pixel = (cpx - 1, cpy - 1)
                    break
                search_dir = (search_dir + 45) % 360
            search_dir = (search_dir + 270) % 360

            if current_pixel == start_pixel:
                break
    return B

# n neighbours
@jit(nopython=True, cache=True)
def get_border_pixels_dense(C):
    h, w = C.shape
    num_classes = np.max(C)
    B = np.zeros((h, w), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            c = C[x, y]
            neighbours = tools.get_valid_neighbours(h, w, x, y)
            if len(neighbours) < 8:
                B[x, y] = c
                continue
            for nx, ny in neighbours:
                if C[nx, ny] != c:
                    B[x, y] = c
                    break

    return B


def get_perimeters(C):
    w2 = 2**0.5

    h, w = C.shape
    num_classes = np.max(C)
    perimeters = np.zeros(num_classes, dtype=np.float64)

    for c in range(1, num_classes):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if C[x, y] == c:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF CLASS 'c'
        search_dir = 0
        current_pixel = start_pixel
        while True:
            cpx, cpy = current_pixel
            while True:
                if search_dir == 0 and cpx > 0 and C[cpx - 1, cpy] == c:
                    current_pixel = (cpx - 1, cpy)
                    perimeters[c] += 1
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and C[cpx - 1, cpy + 1] == c:
                    current_pixel = (cpx - 1, cpy + 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 90 and cpy < w - 1 and C[cpx, cpy + 1] == c:
                    current_pixel = (cpx, cpy + 1)
                    perimeters[c] += 1
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and C[cpx + 1, cpy + 1] == c:
                    current_pixel = (cpx + 1, cpy + 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 180 and cpx < h - 1 and C[cpx + 1, cpy] == c:
                    current_pixel = (cpx + 1, cpy)
                    perimeters[c] += 1
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and C[cpx + 1, cpy - 1] == c:
                    current_pixel = (cpx + 1, cpy - 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 270 and cpy > 0 and C[cpx, cpy - 1] == c:
                    current_pixel = (cpx, cpy - 1)
                    perimeters[c] += 1
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and C[cpx - 1, cpy - 1] == c:
                    current_pixel = (cpx - 1, cpy - 1)
                    perimeters[c] += w2
                    break
                search_dir = (search_dir + 45) % 360
            search_dir = (search_dir + 270) % 360

            if current_pixel == start_pixel:
                break
    return perimeters


# ============= FILTER BANK CLUSTERING ========

def get_filters(sigma_min, sigma_max):
    sl = sigma_min
    sh = sigma_max
    filters = []

    # gaussians

    for sigma in np.linspace(sl, sh, 4):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        filter = np.zeros((N, N), dtype=np.float64)
        offset = N / 2 - 0.5
        two_sigma_sq = 2 * sigma * sigma
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = 1.0 / (np.pi * two_sigma_sq) * np.e ** (-(x * x + y * y) / (two_sigma_sq))
                filter[x_, y_] = G_sigma
        filters += [filter]

    #first, second derivatives

    for sigma in np.linspace(sl, sh, 3):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        offset = N / 2 - 0.5
        two_sigma_sq = 2 * sigma * sigma
        two_pi_sigma_4 = 2 * np.pi * sigma ** 4
        two_pi_sigma_6 = 2 * np.pi * sigma ** 6

        for alpha in np.linspace(0.0, np.pi, 6, endpoint=False):
            filter_first = np.zeros((N, N), dtype=np.float64)
            filter_second = np.zeros((N, N), dtype=np.float64)
            sa = np.sin(alpha)
            ca = np.cos(alpha)
            for x_ in range(N):
                for y_ in range(N):
                    xo = x_ - offset
                    yo = y_ - offset
                    x = ca * xo - sa * yo
                    y = sa * xo + ca * yo
                    # first derivative
                    G_sigma = - x / two_pi_sigma_4 * \
                              np.exp(-(x * x + y * y) / two_sigma_sq)
                    filter_first[x_, y_] = G_sigma
                    # second derivative
                    G_sigma = (x - sigma) * (x + sigma) / two_pi_sigma_6 * \
                              np.exp(-(x * x + y * y) / two_sigma_sq)
                    filter_second[x_, y_] = G_sigma

            filters += [filter_first]
            filters += [filter_second]

    # 8 laplacians

    for sigma in np.linspace(sl, sh, 8):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        filter = np.zeros((N, N), dtype=np.float64)
        offset = N / 2 - 0.5
        f0 = -1 / (np.pi * sigma ** 4)
        two_sigma_sq = 2 * sigma ** 2

        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                xxyy = x * x + y * y
                G_sigma = f0 * (1 - xxyy / two_sigma_sq) * np.exp(-xxyy / two_sigma_sq)
                filter[x_, y_] = G_sigma
        filters += [filter]

    return filters


# filters = get_filters(3)
# for i in range(len(filters)):
#     print((filters[i] * 9 / np.max(np.abs(filters[i]))).astype(np.int))

@jit(int64[:, :](float64[:, :, :], int64, int64, float64), nopython=True, cache=True)
def cluster_features(features, k, iter, th):
    h, w, num_f = features.shape
    means = np.random.uniform(0, 10, (k, num_f)).astype(np.float64)
    sums = np.zeros_like(means)
    counters = np.zeros((k), dtype=np.int64)

    prev_means = np.zeros_like(means)

    print('clustering..')
    for it in range(iter):
        print(it)
        sums *= 0
        counters *= 0

        for x in range(h):
            for y in range(w):
                k_best = -1
                dist_best = 0.0
                feature_vec = features[x, y, :]

                for ki in range(k):
                    dist = np.sum(np.square(feature_vec - means[ki]))
                    if dist < dist_best or k_best == -1:
                        k_best = ki
                        dist_best = dist

                sums[k_best] += feature_vec
                counters[k_best] += 1

        for ki in range(k):
            if counters[ki] > 0:
                means[ki] = sums[ki] / counters[ki]

        if np.max(np.abs(prev_means - means)) < th:
            break
        prev_means = np.copy(means)

    label_map = np.zeros((h, w), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            k_best = -1
            dist_best = 0.0
            feature_vec = features[x, y, :]

            for ki in range(k):
                dist = np.sum(np.square(feature_vec - means[ki]))
                if dist < dist_best or k_best == -1:
                    k_best = ki
                    dist_best = dist

            label_map[x, y] = k_best

    return label_map


def filter_bank_clustering(image, sigma_min, sigma_max, k, th=0.1):
    h, w, d = image.shape

    filters = get_filters(sigma_min, sigma_max)
    num_f = len(filters)
    print('created {} filters'.format(num_f))
    features = np.zeros((h, w, num_f), dtype=np.float64)
    for i in range(num_f):
        print('convolved with {} / {} filters'.format(i, num_f))
        feature_map = convolution.fast_sw_convolution(image, filters[i], convolution.SAME)
        features[:, :, [i]] = feature_map

    features = np.abs(features)

    return cluster_features(features, k, 100, th)
