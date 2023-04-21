import numpy as np
import tools
import convolution
import time
from numba import jit, float64, int64


def tools_test():
    # =============== CONVERSIONS ===================

    C = np.array([[[100, 150, 200], [10, 15, 20]]])
    G = tools.convert_to_1channel(C)
    G3 = tools.convert_to_3channel(G)
    G3_ = np.array([[[150, 150, 150], [15, 15, 15]]])
    print(np.array_equal(G3, G3_))


def test_convolution_speed():
    I = np.random.random((1000, 1000, 3))
    F = np.random.randn(15, 15)

    start = time.time()
    convolution.sliding_window_convolution(I, F)
    print("sliding window convolution {0}".format(time.time() - start))

    start = time.time()
    convolution.sliding_window_convolution_np(I, F)
    print("sliding window convolution np {0}".format(time.time() - start))

    start = time.time()
    convolution.complete_convolution(I, F)
    print("complete convolution {0}".format(time.time() - start))


def test_extension_speed():
    I = np.random.random((3000, 3000, 3))

    start = time.time()
    tools.extend_same(I, 10)
    print("extend same {0}".format(time.time() - start))

    start = time.time()
    tools.extend_with_zeros(I, 10)
    print("extend zero {0}".format(time.time() - start))


def test_non_max_suppression():
    a = np.linspace(1, 27, 27).reshape((3, 3, 3))
    tools.non_max_suppression_3d(a, 1)


def test_non_max_suppression_speed():
    I = np.random.randn(500, 500, 5)

    # start = time.time()
    # _,max_matrix_np = tools.non_max_suppression_np(I,5)
    # print("non max np {0}".format(time.time()-start))

    start = time.time()
    I_mz = np.max(I, 2).reshape(500, 500, 1)
    _, max_matrix = tools.non_max_suppression_3d(I_mz, 2)
    print("non max {0}".format(time.time() - start))

    # print(np.array_equiv(max_matrix,max_matrix_np))


def test_sampling():
    I = (convolution.binomial_filter(12) * 80).astype(int).astype(float).reshape(12, 12, 1)
    I_d = tools.sample_down(I, 2)
    I_dd = tools.sample_down(I_d, 2)
    I_u = tools.sample_up(I_dd, 2)
    I_uu = tools.sample_up(I_u, 2)

    print(I[:, :, 0])
    print(I_d[:, :, 0])
    print(I_dd[:, :, 0])
    print(I_u[:, :, 0])
    print(I_uu[:, :, 0])


# tools_test()
# test_sampling()

@jit(float64[:, :, :](float64[:, :, :], int64), nopython=True,cache = True)
def get_channel(matrix, channel):
    return matrix[:,:,channel:channel+1:]

m = np.random.randn(3,3,3).astype(np.float64)
print(get_channel(m,1))
