
import sys
import os
import cupy as cp

# Update path to find local compiler module
sys.path.append(os.getcwd())


# Device function returning a tuple
from compiler import CUDAProgram

SOURCE = """
def make_tuple(x: int32, y: float32) -> Tuple[int32, float32]:
    return (x, y)

@kernel
def test_tuple_kernel(out_int: Array[int32], out_float: Array[float32]):
    # Test tuple creation and indexing
    t = make_tuple(42, 3.14)
    
    # Verify values
    out_int[0] = t.f0  # Direct field access check (codegen maps t[0] to t.f0 only for constant index)
    # Wait, my codegen maps t[0] to t.f0 if it knows it is a tuple.
    # IRIndex(t, 0)
    
    out_int[0] = t[0]
    out_float[0] = t[1]
    
    # Test nested creation (inline)
    t2: Tuple[int32, int32] = (100, 200)
    out_int[1] = t2[1]
"""

def test_tuple_kernel_wrapper(out_int, out_float):
    prog = CUDAProgram(SOURCE, debug=True)
    kernel = prog.get_kernel('test_tuple_kernel')
    kernel(out_int, out_float)

# Usage in run_test
def run_test():
    print("=== Testing Tuple Support ===")
    
    out_int = cp.zeros(10, dtype=cp.int32)
    out_float = cp.zeros(10, dtype=cp.float32)

    # Compile and run
    # Using wrapper logic inline
    prog = CUDAProgram(SOURCE, debug=True)
    prog.run('test_tuple_kernel', out_int, out_float)
    
    res_int = cp.asnumpy(out_int)
    res_float = cp.asnumpy(out_float)
    
    print(f"Int results: {res_int[:2]}")
    print(f"Float results: {res_float[:1]}")
    
    assert res_int[0] == 42
    assert abs(res_float[0] - 3.14) < 1e-5
    assert res_int[1] == 200
    
    print("✅ Tuple test passed!")

if __name__ == "__main__":
    run_test()
