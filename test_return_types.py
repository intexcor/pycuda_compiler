
from compiler import cuda_compile, int32, float32, Array, List
import cupy as cp
import numpy as np

# Mock classes to support type hints without full imports if needed
class Tuple:
    def __class_getitem__(cls, item): return f'Tuple[{item}]'

@cuda_compile
def kernel_return_scalar(out: Array[int32], n: int32):
    if n > 10:
        return
    out[0] = 42

@cuda_compile(debug=True)
def test_returns(out: Array[int32]):
    # Test scalar return
    val = kernel_return_scalar(out, 5) # This call inside kernel not supported yet? 
    # Wait, kernel_return_scalar is a kernel (global). Can't call global from global.
    # It needs to be a device function.
    pass

# We need a device function to test return values.
# The user wants to return FROM a function.
# If they return from KERNEL, they can only return void (early exit).
# If they return from DEVICE function, they can return values.

from compiler import compile_module

source = """
from typing import Tuple

@device
def get_magic_number() -> int32:
    return 42

@device
def get_tuple() -> Tuple[int32, float32]:
    return (123, 4.56)

@device
def get_tuple_with_array() -> Tuple[int32, Array[float32]]:
    arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    # arr is ScopedArray here.
    # returning (int, arr) should call arr.release()
    return (999, arr)

# String not fully exposed in python parser as a type we can import?
# But literals work.

@kernel
def test_kernel(out: Array[int32], out_float: Array[float32]):
    n = get_magic_number()
    out[0] = n
    
    t = get_tuple()
    out[1] = t[0]
    out_float[0] = t[1]
    
    # Test tuple with array
    t2 = get_tuple_with_array()
    out[2] = t2[0]
    # t2[1] is now a raw pointer (float*) because Tuple struct holds raw pointers
    # We should access it carefully. t2[1] is NOT a ScopedArray in the kernel (it's a field in a struct)
    # The kernel needs to WRAP it in ScopedArray if it wants to own it?
    # OR, if it's just using it, it's fine.
    # BUT who deletes it?
    # The returned tuple has a pointer.
    # The pointer is NOT owned by anyone in `test_kernel`.
    # ScopedArray wrapper is needed!
    # Currently my compiler does NOT automatically wrap received pointers into ScopedArray.
    # That is a hole in safety if we return owning pointers inside structs.
    # However, for this test, let's just use it to verify data correctness.
    # We can manually delete it to be safe, or accept leakage in test.
    
    ptr = t2[1]
    out_float[1] = ptr[1] # Should be 2.2
    
    del ptr # Manual cleanup
"""

if __name__ == "__main__":
    program = compile_module(source)
    
    out = np.zeros(5, dtype=np.int32)
    out_float = np.zeros(5, dtype=np.float32)
    
    program.run('test_kernel', out, out_float)
    
    print("Out:", out)
    print("Out Float:", out_float)
    
    assert out[0] == 42
    assert out[1] == 123
    assert out[2] == 999
    assert abs(out_float[0] - 4.56) < 1e-5
    assert abs(out_float[1] - 2.2) < 1e-5
    
    print("Verification Passed!")
