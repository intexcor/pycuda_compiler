from compiler import cuda_compile, Array, List, float32, int32
import numpy as np
import cupy as cp

def test_list_append():
    print("\n=== Testing List Append ===")
    
    @cuda_compile
    def filter_positive(data: Array[float32], result: List[float32]):
        for i in range(len(data)):
            val = data[i]
            if val > 0.0:
                result.append(val)
    
    # Data
    N = 1000
    data = np.random.randn(N).astype(np.float32)
    positive_count = np.sum(data > 0)
    
    # Result list buffer
    # Capacity = N (worst case)
    res_buf = cp.zeros(N, dtype=cp.float32)
    res_size = cp.zeros(1, dtype=cp.int32)
    
    # Run
    # Arguments: (data, data_size), (res_buf, res_size)
    # The List argument in python call corresponds to the tuple (buffer, size) as we implemented in compiler.py
    filter_positive(
        cp.asarray(data), 
        (res_buf, res_size)
    )
    
    # Verify
    count = int(res_size[0])
    print(f"Original positive count: {positive_count}")
    print(f"Kernel result count:     {count}")
    
    if count != positive_count:
        print("❌ Counts mismatch!")
        exit(1)
        
    # Check values
    # Note: Order is not guaranteed due to parallel execution + atomicAdd
    res_cpu = cp.asnumpy(res_buf[:count])
    res_cpu.sort()
    
    expected = data[data > 0]
    expected.sort()
    
    if np.allclose(res_cpu, expected):
        print("✅ Values match (sorted)!")
    else:
        print("❌ Values mismatch!")
        print("Result:", res_cpu[:10])
        print("Expect:", expected[:10])
        exit(1)

def test_list_len():
    print("\n=== Testing List Len ===")
    
    @cuda_compile
    def get_len(lst: List[int32], out: Array[int32]):
        out[0] = lst.len()

    N = 123
    lst_buf = cp.zeros(N, dtype=cp.int32)
    lst_size = cp.array([N], dtype=cp.int32)
    out = cp.zeros(1, dtype=cp.int32)
    
    get_len(
        (lst_buf, lst_size),
        out,
        grid=(1,), block=(1,)
    )
    
    res = int(out[0])
    print(f"Expected len: {N}")
    print(f"Kernel len:   {res}")
    
    if res != N:
        print("❌ Len mismatch!")
        exit(1)
    print("✅ Len matches!")

def test_list_read():
    print("\n=== Testing List Read ===")
    @cuda_compile
    def read_idx(lst: List[int32], idx: int32, out: Array[int32]):
        out[0] = lst[idx]

    data = np.array([10, 20, 30], dtype=np.int32)
    lst_buf = cp.asarray(data)
    lst_size = cp.array([3], dtype=cp.int32)
    out = cp.zeros(1, dtype=cp.int32)
    
    read_idx(
        (lst_buf, lst_size),
        1,
        out,
        grid=(1,), block=(1,)
    )
    
    res = int(out[0])
    print("Expected val at idx 1: 20")
    print(f"Kernel val:            {res}")
    
    if res != 20:
        print("❌ Read mismatch!")
        exit(1)
    print("✅ Read matches!")

def test_list_access():
    print("\n=== Testing List Sum Loop ===")
    
    @cuda_compile
    def sum_list(lst: List[int32], out: Array[int32]):
        s: int32 = 0
        n = lst.len()
        for i in range(n):
            s += lst[i]
        out[0] = s

    # Fill list
    N = 100
    data = np.arange(N, dtype=np.int32)
    expected_sum = np.sum(data)
    
    lst_buf = cp.asarray(data)
    lst_size = cp.array([N], dtype=cp.int32)
    
    out = cp.zeros(1, dtype=cp.int32)
    
    # Run with 1 thread to avoid race condition on out[0]
    # (Though theoretical race of writing same value should be safe)
    sum_list(
        (lst_buf, lst_size),
        out,
        grid=(1,), block=(1,)
    )
    
    res = int(out[0])
    print(f"Expected sum: {expected_sum}")
    print(f"Kernel sum:   {res}")
    
    if res == expected_sum:
        print("✅ Sum matches!")
    else:
        print("❌ Sum mismatch!")
        exit(1)

if __name__ == "__main__":
    try:
        test_list_append()
        test_list_len()
        test_list_read()
        test_list_access()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
