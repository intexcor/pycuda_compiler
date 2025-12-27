
import cupy as cp
from compiler import CUDAProgram

SOURCE = """
@kernel
def test_dict(out_keys: Array[int32], out_vals: Array[int32], check_res: Array[int32]):
    # Local dictionary (capacity 256 by default)
    counts: Dict[int32, int32]
    
    # Insert values
    counts[10] = 100
    counts[20] = 200
    counts[30] = 300
    
    # Update value
    counts[10] += 1  # 101
    
    # Check "in" operator
    if 10 in counts:
        out_keys[0] = 10
        out_vals[0] = counts[10]
        check_res[0] = 1
    else:
        check_res[0] = 0
        
    if 20 in counts:
        out_keys[1] = 20
        out_vals[1] = counts[20]
        check_res[1] = 1
        
    if 99 in counts:
        check_res[2] = 1
    else:
        check_res[2] = 0 # Correct
"""

def run_test():
    print("=== Testing Dict Support ===")
    
    out_keys = cp.zeros(3, dtype=cp.int32)
    out_vals = cp.zeros(3, dtype=cp.int32)
    check_res = cp.zeros(3, dtype=cp.int32)
    
    # Compile and run
    prog = CUDAProgram(SOURCE, debug=True)
    prog.run('test_dict', out_keys, out_vals, check_res)
    
    res_keys = cp.asnumpy(out_keys)
    res_vals = cp.asnumpy(out_vals)
    res_check = cp.asnumpy(check_res)
    
    print("Keys:", res_keys)
    print("Vals:", res_vals)
    print("Check:", res_check)
    
    # Expected: 
    # Index 0 (key 10): val 101, check 1
    # Index 1 (key 20): val 200, check 1
    # Index 2 (key 99): check 0
    
    assert res_check[0] == 1
    assert res_vals[0] == 101
    
    assert res_check[1] == 1
    assert res_vals[1] == 200
    
    assert res_check[2] == 0
    
    print("✅ Dict test passed!")

if __name__ == '__main__':
    run_test()
