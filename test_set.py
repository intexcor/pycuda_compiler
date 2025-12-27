
import cupy as cp
import numpy as np
from compiler import CUDAProgram

SOURCE = """
@kernel
def test_set(out_check: Array[int32]):
    # Local set (capacity 256)
    s: Set[int32]
    
    s.add(10)
    s.add(20)
    
    # Check "in"
    if 10 in s:
        out_check[0] = 1
    else:
        out_check[0] = 0
        
    # Check "not in"
    if 30 not in s:
        out_check[1] = 1
    else:
        out_check[1] = 0
        
    if 20 in s:
        out_check[2] = 1
    else:
        out_check[2] = 0
"""

def run_test():
    print("=== Testing Set Support ===")
    
    out_check = cp.zeros(3, dtype=cp.int32)
    
    # Compile and run
    prog = CUDAProgram(SOURCE, debug=True)
    prog.run('test_set', out_check)
    
    res = cp.asnumpy(out_check)
    print("Check:", res)
    
    assert res[0] == 1
    assert res[1] == 1
    assert res[2] == 1
    
    print("✅ Set test passed!")

if __name__ == '__main__':
    run_test()
