import cupy as cp
from compiler import cuda_compile, Array, float32
from math import sqrt

@cuda_compile
def square_root_kernel(data: Array[float32], result: Array[float32]):
    # Automatic parallelization handles the loop distributions!
    for i in range(len(data)):
        result[i] = sqrt(data[i])

if __name__ == "__main__":
    # Create data on GPU
    data = cp.array([1.0, 4.0, 9.0, 16.0], dtype=cp.float32)
    result = cp.zeros_like(data)

    # Run kernel
    square_root_kernel(data, result)

    print(result) # [1. 2. 3. 4.]