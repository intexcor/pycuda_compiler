
import cupy as cp
from compiler import CUDAProgram

SOURCE = """
@kernel
def test_string(out_len: Array[int32], out_eq: Array[int32]):
    s1 = "hello"
    s2 = "world"
    s3 = "hello"
    
    out_len[0] = len(s1)
    # Check equality
    if s1 == s3:
        out_eq[0] = 1
    else:
        out_eq[0] = 0
        
    if s1 == s2:
        out_eq[1] = 1
    else:
        out_eq[1] = 0
        
    # Check modification (optional, if mutable)
    # s1 is String struct, mutable local variable
"""

def run_test():
    print("=== Testing String Support ===")
    
    out_len = cp.zeros(1, dtype=cp.int32)
    out_eq = cp.zeros(2, dtype=cp.int32)
    
    # Compile and run
    prog = CUDAProgram(SOURCE, debug=True)
    prog.run('test_string', out_len, out_eq)
    
    res_len = cp.asnumpy(out_len)
    res_eq = cp.asnumpy(out_eq)
    
    print("Length:", res_len)
    print("Equality:", res_eq)
    
    assert res_len[0] == 5
    assert res_eq[0] == 1
    assert res_eq[1] == 0
    
    print("✅ String test passed!")

if __name__ == '__main__':
    run_test()
