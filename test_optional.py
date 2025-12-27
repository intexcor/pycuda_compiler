
import cupy as cp
import numpy as np
from compiler import CUDAProgram

SOURCE = """
# Pseudo-imports for types
# In real usage these would be imported from pycuda_compiler
# but here we just rely on the parser recognizing names.

def Some(x: float32) -> Optional[float32]:
    pass # Intrinsic

def safe_div(a: float32, b: float32) -> Optional[float32]:
    if b == 0.0:
        return None
    else:
        return Some(a / b)

@kernel
def test_optional_kernel(a: Array[float32], b: Array[float32], out: Array[float32], valid: Array[int32]):
    for i in range(len(a)):
        res = safe_div(a[i], b[i])
        if res.has_value:
             out[i] = res.value
             valid[i] = 1
        else:
             out[i] = 0.0
             valid[i] = 0
"""

def run_test():
    print("=== Testing Optional Support ===")
    
    # Data
    N = 10
    a = cp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=cp.float32)
    b = cp.array([2.0,  5.0,  0.0,  4.0,  0.0,  6.0,  7.0,  0.0,  9.0,  10.0], dtype=cp.float32)
    # Expected:   5.0,  4.0,  Inv,  10.0, Inv,  10.0, 10.0, Inv,  10.0, 10.0
    
    out = cp.zeros(N, dtype=cp.float32)
    valid = cp.zeros(N, dtype=cp.int32)
    
    # Compile and run
    prog = CUDAProgram(SOURCE, debug=True)
    print("=== Generated CUDA Code ===")
    prog.show_cuda()
    prog.run('test_optional_kernel', a, b, out, valid)
    
    res_out = cp.asnumpy(out)
    res_valid = cp.asnumpy(valid)
    
    print("Input A:", a)
    print("Input B:", b)
    print("Output:", res_out)
    print("Valid:", res_valid)
    
    expected_valid = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int32)
    expected_out = np.zeros(N, dtype=np.float32)
    mask = expected_valid == 1
    expected_out[mask] = cp.asnumpy(a)[mask] / cp.asnumpy(b)[mask]
    
    np.testing.assert_array_equal(res_valid, expected_valid)
    np.testing.assert_allclose(res_out, expected_out, atol=1e-5)
    
    print("✅ Optional test passed!")

if __name__ == '__main__':
    run_test()
