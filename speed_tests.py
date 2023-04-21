import time
import numpy as np
from numba import jit,float64,float32



@jit(float64(float64,float64),nopython = True,cache = True)
def a(a,b):
    return a**b


@jit(float64(float64,float64),nopython = True,cache = True)
def b(a,b):
    return np.power(a,b)

def test_jit_functions():

    start = time.time()
    for i in range(1000000):
        v1 = np.random.random()
        v2 = np.random.random()
        a(v1,v2)
    print("a {0}".format(time.time() - start))


    start = time.time()
    for i in range(1000000):
        v1 = np.random.random()
        v2 = np.random.random()
        b(v1,v2)
    print("b {0}".format(time.time() - start))

test_jit_functions()