import numpy as np
import tools
from numba import jit, float64, int64, boolean

GROUPNAME = 'Morphological'

EROSION = 'Erosion'
DILATATION = 'Dilation'
OPENING = 'Opening'
CLOSING = 'Closing'

DISTANCE_MAP = 'Distance Map'

OPS = (EROSION, DILATATION, OPENING, CLOSING, DISTANCE_MAP)

ERODE = 0
DILATE = 1


def apply(in_image, operation, p1, p2, p3, p4):
    binary_image = tools.convert_to_binary(in_image)
    h, w, d = in_image.shape

    # ============= APPLY OPERATION ===============

    if operation == EROSION:
        iterations = int(p3)
        N = int(p2)
        if p1 == 'circle' or p1 == 'c':
            SE = get_structure_element_circle(N)
        elif p1 == 'square' or p1 == 's':
            SE = get_structure_element_square(N)
        else:
            raise ValueError
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, ERODE)
        return tools.normalize(binary_image.reshape(h, w, 1).astype(np.float64))

    elif operation == DILATATION:
        iterations = int(p3)
        N = int(p2)
        if p1 == 'circle' or p1 == 'c':
            SE = get_structure_element_circle(N)
        elif p1 == 'square' or p1 == 's':
            SE = get_structure_element_square(N)
        else:
            raise ValueError
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, DILATE)
        return tools.normalize(binary_image.reshape(h, w, 1).astype(np.float64))

    elif operation == OPENING:
        iterations = int(p3)
        N = int(p2)
        if p1 == 'circle' or p1 == 'c':
            SE = get_structure_element_circle(N)
        elif p1 == 'square' or p1 == 's':
            SE = get_structure_element_square(N)
        else:
            raise ValueError
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, ERODE)
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, DILATE)
        return tools.normalize(binary_image.reshape(h, w, 1).astype(np.float64))

    elif operation == CLOSING:
        iterations = int(p3)
        N = int(p2)
        if p1 == 'circle' or p1 == 'c':
            SE = get_structure_element_circle(N)
        elif p1 == 'square' or p1 == 's':
            SE = get_structure_element_square(N)
        else:
            raise ValueError
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, DILATE)
        for i in range(iterations):
            binary_image = apply_operation(binary_image, SE, ERODE)
        return tools.normalize(binary_image.reshape(h, w, 1).astype(np.float64))

    elif operation == DISTANCE_MAP:
        if p1 == 'manhattan' or p1 == 'm':
            distance_map = distance_transform_manhattan(binary_image)
        else:
            N = int(p2)
            distance_map = distance_transform_euclidian(binary_image, N)
        return tools.normalize(distance_map.reshape(h, w, 1).astype(np.float64))


@jit(boolean[:, :](int64), nopython=True, cache=True)
def get_structure_element_circle(D):
    if D % 2 == 0:
        D += 1
    cx = D / 2
    cy = D / 2
    d_max = (D / 2) - 0.3

    SE = np.zeros((D, D), dtype=np.bool_)
    for x in range(D):
        for y in range(D):
            xc = x + 0.5
            yc = y + 0.5
            dx = cx - xc
            dy = cy - yc
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= d_max:
                SE[x, y] = 1

    return SE


@jit(boolean[:, :](int64), nopython=True, cache=True)
def get_structure_element_square(N):
    if N % 2 == 0:
        N += 1
    return np.ones((N, N), dtype=np.bool_)


@jit(boolean[:, :](boolean[:, :], boolean[:, :], int64), nopython=True, cache=True)
def apply_operation(binary_image, structural_element, operation):
    fs = structural_element.shape[0]
    hfs = int(fs / 2)

    h, w = binary_image.shape
    out_image = np.zeros((h, w), dtype=np.bool_)

    for x_a in range(h):
        for y_a in range(w):
            im_x_low = max(0, x_a - hfs)
            im_x_high = min(h, x_a + 1 + hfs)
            im_y_low = max(0, y_a - hfs)
            im_y_high = min(w, y_a + 1 + hfs)
            im_area = binary_image[im_x_low:im_x_high, im_y_low:im_y_high].astype(np.int64)

            se_x_low = max(0, hfs - x_a)
            se_x_high = min(fs, h - x_a + hfs)
            se_y_low = max(0, hfs - y_a)
            se_y_high = min(fs, w - y_a + hfs)
            se_area = structural_element[se_x_low:se_x_high, se_y_low:se_y_high].astype(np.int64)

            if operation == DILATE:
                out_image[x_a, y_a] = np.max(np.logical_and(im_area, se_area))
                # print(np.max(im_area * se_area))
                # out_image[x_a, y_a] = np.max(im_area * se_area)
            elif operation == ERODE:
                # print(np.min(im_area * se_area))
                out_image[x_a, y_a] = 1 - np.max(np.logical_and(1 - im_area, se_area))
                # out_image[x_a, y_a] = np.min(im_area * se_area)

    return out_image


@jit(int64[:, :](boolean[:, :]), nopython=True, cache=True)
def distance_transform_manhattan(binary_image):
    h, w = binary_image.shape

    border_value = 0

    FP = np.zeros((h, w), dtype=np.int64)
    for x in range(h):
        for y in range(w):
            if binary_image[x, y]:
                north_neighbour = border_value
                if x > 0:
                    north_neighbour = 1 + FP[x - 1, y]

                west_neighbour = border_value
                if y > 0:
                    west_neighbour = 1 + FP[x, y - 1]

                FP[x, y] = min(north_neighbour, west_neighbour)

    BP = np.zeros((h, w), dtype=np.int64)
    for x in range(h - 1, -1, -1):
        for y in range(w - 1, -1, -1):
            if binary_image[x, y]:
                south_neighbour = border_value
                if x < h - 1:
                    south_neighbour = 1 + BP[x + 1, y]

                east_neighbour = border_value
                if y < w - 1:
                    east_neighbour = 1 + BP[x, y + 1]

                BP[x, y] = min(FP[x, y], min(south_neighbour, east_neighbour))

    return BP


@jit(float64[:, :](boolean[:, :], int64), nopython=True, cache=True)
def distance_transform_euclidian(binary_image, N):
    h, w = binary_image.shape

    FP_values = np.zeros((N + 1, N + 1), dtype=np.float64)
    BP_values = np.zeros((N + 1, N + 1), dtype=np.float64)
    FPI_values = np.zeros((N + 1, N + 1), dtype=np.float64)
    BPI_values = np.zeros((N + 1, N + 1), dtype=np.float64)
    for x in range(N + 1):
        for y in range(x, N + 1):
            if x == 0 and y == 0:
                v = h * w
            else:
                v = np.sqrt(x * x + y * y)
            FP_values[N - x, N - y] = v
            BP_values[x, y] = v
            FPI_values[N - x, y] = v
            BPI_values[x, N - y] = v

            if x != y:
                FP_values[N - y, N - x] = v
                BP_values[y, x] = v
                FPI_values[N - y, x] = v
                BPI_values[y, N - x] = v

    # FP_values[N, N] = h * w
    # BP_values[0, 0] = h * w
    #
    # FP_values[0,0] = 0.0
    # FP_values[1,0] = 1.0
    # FP_values[0,1] = 1.0
    # FP_values[1,1] = 1.4

    FP = np.zeros((h + 2 * N, w + 2 * N), dtype=np.float64)
    BP = np.zeros((h + 2 * N, w + 2 * N), dtype=np.float64)
    FPI = np.zeros((h + 2 * N, w + 2 * N), dtype=np.float64)
    BPI = np.zeros((h + 2 * N, w + 2 * N), dtype=np.float64)

    for x in range(N, N + h):
        for y in range(N, N + w):
            if binary_image[x - N, y - N]:
                neighbours = FP[x - N:x + 1, y - N:y + 1]
                FP[x, y] = np.min(neighbours + FP_values)

    for x in range(h + N - 1, N - 1, -1):
        for y in range(w + N - 1, N - 1, -1):
            if binary_image[x - N, y - N]:
                neighbours = BP[x:x + 1 + N, y:y + 1 + N]
                BP[x, y] = min(FP[x, y], np.min(neighbours + BP_values))

    for x in range(N, N + h):
        for y in range(w + N - 1, N - 1, -1):
            if binary_image[x - N, y - N]:
                neighbours = FPI[x - N:x + 1, y:y + 1 + N]
                FPI[x, y] = min(BP[x, y], np.min(neighbours + FPI_values))

    for x in range(h + N - 1, N - 1, -1):
        for y in range(N, N + w):
            if binary_image[x - N, y - N]:
                neighbours = BPI[x:x + 1 + N, y - N:y + 1]
                BPI[x, y] = min(FPI[x, y], np.min(neighbours + BPI_values))

    return BPI[N:N + h, N:N + w]
