
import cupy as cp

print(f"CuPy version: {cp.__version__}")

code = r'''
#include <cuda_runtime.h>
#include <math.h>

// Thread indexing helpers
#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define GRID_STRIDE (blockDim.x * gridDim.x)



__global__ void vector_add(float* a, int _size_a, float* b, int _size_b, float* c, int _size_c) {
    int i;
    for (int i = THREAD_ID; i < _size_a; i += GRID_STRIDE) {
        c[i] = (a[i] + b[i]);
    }
}

'''

try:
    print("Compiling module...")
    module = cp.RawModule(code=code, backend='nvrtc', options=('--std=c++14',), name_expressions=['vector_add'])
    print("Module compiled.")
    
    try:
        ptx = module.get_ptx()
        if ptx:
            print("PTX dump:")
            print(ptx.decode() if isinstance(ptx, bytes) else str(ptx))
        else:
            print("PTX is None")
    except Exception as e:
        print(f"Could not get PTX: {e}")

    print("Getting function 'vector_add'...")
    kernel = module.get_function('vector_add')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
