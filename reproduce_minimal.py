
import cupy as cp

code = r'''


#include <math.h>

extern "C" {
    __global__ void vector_add(float* a, float* b, float* c) {
        c[0] = a[0] + b[0];
    }
}

'''

try:
    print("Compiling module...")
    module = cp.RawModule(code=code, backend='nvrtc', options=('--std=c++14',))
    print("Module compiled.")
    
    print("Getting function 'vector_add'...")
    kernel = module.get_function('vector_add')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
