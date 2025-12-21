
import cupy as cp
import numpy as np

code = r'''
extern "C" __global__ void test_math(float* a) {
    a[0] = sqrt(a[0]);
}
'''

try:
    print("Compiling module...")
    module = cp.RawModule(code=code, backend='nvrtc', options=('--std=c++14',))
    print("Module compiled.")
    
    print("Getting function 'test_math'...")
    kernel = module.get_function('test_math')
    print("Success!")
    
    # Run it
    a = cp.array([4.0], dtype=np.float32)
    kernel((1,), (1,), (a,))
    print(f"Result: {a}")
except Exception as e:
    print(f"Error: {e}")
