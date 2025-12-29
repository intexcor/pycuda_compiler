
from compiler import cuda_compile as compile, int32, float32, Array
try:
    import numpy as np
except ImportError:
    class MockNumpy:
        int32 = 'int32'
        float32 = 'float32'
        def zeros(self, *args, **kwargs): pass
        def empty(self, *args, **kwargs): pass
    np = MockNumpy()


def test_numpy_creation():
    @compile
    def kernel_numpy(out: Array[int32], n: int32):
        # Stack/Local array (Fixed size) -> Optimization ideally
        # a = np.array([1, 2, 3]) 
        # out[0] = a[0]
        
        # Dynamic allocation
        b = np.zeros(10, dtype=np.int32)
        b[0] = 42
        out[1] = b[0]

        # Test early return (scoped arrays should be cleaned up)
        if n < 0:
            return

        c = np.empty(5, dtype=np.float32)
        c[0] = 3.14
        out[2] = int(c[0])

        e = np.array([10, 20, 30], dtype=np.int32)
        out[3] = e[1]
        
        # Automatic deallocation should happen here

    kernel_numpy.show_cuda()

    # Prepare arguments
    # import cupy as cp (Not needed now if using numpy!)
    # out = np.zeros(5, dtype=np.int32)
    
    out = np.zeros(5, dtype=np.int32)
    
    print("Running kernel with NumPy host array...")
    kernel_numpy(out, 10)
    
    print("Kernel finished. Result should be updated in-place.")
    print("Result out:", out)

    # Verification
    # out[1] = 42 (from b[0])
    # out[2] = 3 (from c[0]=3.14)
    # out[3] = 20 (from e[1]=20)
    
    expected = [0, 42, 3, 20, 0]
    # out is already numpy array
    
    print(f"Expected: {expected}")
    print(f"Actual:   {out.tolist()}")
    
    assert out[1] == 42
    assert out[2] == 3
    assert out[3] == 20
    print("Verification Passed!")

if __name__ == "__main__":
    try:
        test_numpy_creation()
        print("Test Complete")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {e}")
