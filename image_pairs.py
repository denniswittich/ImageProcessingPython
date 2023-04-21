import numpy as np
import tools
import convolution
import detection

GROUPNAME = 'Image Pairs'

SIFT_MATCH = 'D.O.G. + Sift'
DIFFERENCE = 'Difference'

OPS = (SIFT_MATCH, DIFFERENCE)


def apply(image_1, image_2, operation, p1, p2, p3, p4):
    if operation == SIFT_MATCH:
        gray_in_image_1 = tools.convert_to_1channel(image_1)
        gray_in_image_2 = tools.convert_to_1channel(image_2)

        octaves = int(float(p1))
        num_per_octave = int(float(p2))
        num_best = int(float(p3))
        threshold = float(p4)

        features_1 = detection.difference_of_gaussian(gray_in_image_1, octaves, num_per_octave, threshold, num_best)
        features_2 = detection.difference_of_gaussian(gray_in_image_2, octaves, num_per_octave, threshold, num_best)

        descriptors_1, orientations_1 = sift_descriptor(gray_in_image_1, features_1)
        descriptors_2, orientations_2 = sift_descriptor(gray_in_image_2, features_2)

        result = create_match_image(image_1, features_1, orientations_1, descriptors_1,
                                    image_2, features_2, orientations_2, descriptors_2)
        return result


def sift_descriptor(image, features):
    num_features = features.shape[0]
    N_o = 5
    rad2deg = 180 / np.pi
    deg2rad = np.pi / 180

    ### MAIN FEATURE ORIENTATION

    gauss_o = convolution.gauss_filter_N(N_o / 6, N_o)
    orientations = np.zeros((num_features), dtype=np.float64)
    for i in range(num_features):
        bins = np.zeros((36), dtype=float)
        x, y, sigma = features[i]
        window_width = 10 * sigma
        coordinates = tools.patch_coordinates(x, y, N_o, window_width)

        fdgx = convolution.derivative_of_gaussian(sigma, 0)
        fdgy = convolution.derivative_of_gaussian(sigma, 1)

        for xc in range(N_o):
            for yc in range(N_o):
                xp, yp = coordinates[xc, yc, :]  # resampled coordinates of patch
                w = gauss_o[xc, yc]

                dx = convolution.convolve_at(image, fdgx, xp, yp, 0)
                dy = convolution.convolve_at(image, fdgy, xp, yp, 0)

                amplitude = np.sqrt(dx * dx + dy * dy)
                orientation = np.arctan2(dy, dx) * rad2deg
                if orientation < 0:
                    orientation += 360

                bin_index = int(orientation // 36)
                bins[bin_index] += w * amplitude
        orientations[i] = (np.argmax(bins) * 36.0 + 5) * deg2rad

    ### FEATURE DESCRIPTORS

    N_d = 16
    gauss_d = convolution.gauss_filter_N(N_d / 2, N_d)

    descriptors = np.zeros((num_features, 128), dtype=np.float64)
    for i in range(num_features):
        bins_d = np.zeros((4, 4, 8))
        x, y, sigma = features[i]
        # print(sigma)
        window_width = max(16, 8 * sigma)  # 6 also works well
        coordinates = tools.patch_coordinates_rotated(x, y, N_d, window_width, orientations[i])
        fdgx = convolution.derivative_of_gaussian(sigma, 0)
        fdgy = convolution.derivative_of_gaussian(sigma, 1)

        for xc in range(N_d):
            for yc in range(N_d):
                xp, yp = coordinates[xc, yc, :]  # resampled coordinates of patch
                w = gauss_d[xc, yc]

                dx = convolution.convolve_at(image, fdgx, xp, yp, 0)
                dy = convolution.convolve_at(image, fdgy, xp, yp, 0)

                amplitude = np.sqrt(dx * dx + dy * dy)
                orientation = (np.arctan2(dy, dx) - orientations[i]) * rad2deg
                if orientation < 0:
                    orientation += 360

                bin_index = int(orientation // 45)
                bins_d[xc // 4, yc // 4, bin_index] += amplitude  # * w

        for x_ in range(4):
            for y_ in range(4):
                bins_d[x_, y_, :] /= np.sum(bins_d[x_, y_, :])

        descriptors[i, :] = bins_d.reshape((128))

    return descriptors, orientations


def create_match_image(image_1, features_1, orientations_1, descriptors_1, image_2, features_2, orientations_2,
                       descriptors_2):
    num_features_1 = features_1.shape[0]
    num_features_2 = features_2.shape[0]

    image_1 = tools.convert_to_3channel(image_1)
    image_2 = tools.convert_to_3channel(image_2)
    h1, w1, _ = image_1.shape
    h2, w2, _ = image_2.shape

    best_matches_from_i = np.ones((num_features_1), dtype=np.int64) * -1
    best_matches_from_i_values = np.zeros((num_features_1), dtype=np.float64)
    best_matches_from_j = np.ones((num_features_2), dtype=np.int64) * -1

    for i in range(num_features_1):
        d_0 = descriptors_1[i, :]
        best_dist = 10000.0
        best_j = None
        for j in range(len(features_2)):
            d_1 = descriptors_2[j, :]
            diff = d_0 - d_1
            dist = diff.T.dot(diff)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if not best_j is None:
            best_matches_from_i[i] = best_j
            best_matches_from_i_values[i] = best_dist

    for j in range(len(features_2)):
        d_1 = descriptors_2[j, :]
        best_dist = 10000.0
        best_i = None
        for i in range(len(features_1)):
            d_0 = descriptors_1[i, :]
            diff = d_1 - d_0
            dist = diff.T.dot(diff)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if not best_i is None:
            best_matches_from_j[j] = best_i

    ### REMOVE MATCHES where best_i[j] != i
    num_double_match = 0
    for i in range(len(features_1)):
        j = best_matches_from_i[i]
        if j == -1 or best_matches_from_j[j] != i:
            best_matches_from_i_values[i] = 10000.0
        else:
            num_double_match += 1

    num_best_matches = 10
    num_best_matches = min(num_best_matches, num_double_match)

    vectors = np.zeros((num_best_matches, 4), dtype=np.int64)
    directions1 = np.zeros((num_best_matches, 4), dtype=np.float64)
    circles1 = np.zeros((num_best_matches, 3), dtype=np.float64)
    directions2 = np.zeros((num_best_matches, 4), dtype=np.float64)
    circles2 = np.zeros((num_best_matches, 3), dtype=np.float64)

    sq2 = np.sqrt(2)
    for v in range(num_best_matches):
        b_i = np.argmin(best_matches_from_i_values)
        best_matches_from_i_values[b_i] = 10000.0
        b_j = best_matches_from_i[b_i]

        x1, y1 = features_1[b_i, :2]
        r1 = features_1[b_i, 2] * sq2
        o1 = orientations_1[b_i]

        x2, y2 = features_2[b_j, :2]
        y2 += w1
        r2 = features_2[b_j, 2] * sq2
        o2 = orientations_2[b_j]

        vectors[v, :] = ([x1, y1, x2, y2])

        circles1[v] = ([x1, y1, r1])
        dx1 = np.cos(o1) * 3 * r1
        dy1 = np.sin(o1) * 3 * r1
        directions1[v, :] = ([x1, y1, x1 + dx1, y1 + dy1])

        circles2[v] = ([x2, y2, r2])
        dx2 = np.cos(o2) * 3 * r2
        dy2 = np.sin(o2) * 3 * r2
        directions2[v, :] = ([x2, y2, x2 + dx2, y2 + dy2])

    whole = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float64)
    whole[:h1, :w1, :] = image_1
    whole[:h2, w1:, :] = image_2

    tools.draw_vectors(whole, vectors.astype(np.float64))
    tools.draw_circles(whole, circles1)
    tools.draw_circles(whole, circles2)
    tools.draw_red_vectors(whole, directions1)
    tools.draw_red_vectors(whole, directions2)
    return whole
