
import cupy as cp

code = r'''
#include <cuda_runtime.h>
#include <math.h>

__global__ void vector_add(float* a, float* b, float* c) {
    c[0] = a[0] + b[0];
}
'''

try:
    print("Compiling module...")
    module = cp.RawModule(code=code, backend='nvrtc', options=('--std=c++14',), name_expressions=['vector_add'])
    print("Module compiled.")
    
    print("Getting function 'vector_add'...")
    # When using name_expressions, we don't access via get_function nicely usually?
    # Actually RawModule.get_function should work if name_expressions was used.
    # But usually one accesses it via the mangled name, OR cupy handles it.
    
    # CuPy documentation says:
    # "The function retrieval API is consistent with that of the standard module."
    kernel = module.get_function('vector_add')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
