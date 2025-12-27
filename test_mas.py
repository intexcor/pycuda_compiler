
import sys
import os
import numpy as np
import time

# Update path to find local compiler module
sys.path.append(os.getcwd())

try:
    import cupy as cp
    from compiler import cuda_compile, Tensor, float32, int32, kernel
except ImportError as e:
    print(f"Cupy or Compiler not found: {e}")
    sys.exit(1)

# Pure Python Reference
def maximum_path_python(paths, values, t_ys, t_xs):
    """Pure Python reference implementation of Monotonic Alignment Search"""
    b = paths.shape[0]
    max_neg_val = -1e9
    
    for i in range(b):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]
        
        v_prev = v_cur = 0.0
        index = t_x - 1
        
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                
                value[y, x] += max(v_prev, v_cur)
                
        # Backtrack
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index = index - 1

# CUDA Kernel
# This uses the EXACT logic as the python reference (mostly), 
# effectively compiling "one function".
# We only add type hints.

@cuda_compile(debug=False)
def maximum_path_jit(paths: Tensor[int32, 3], values: Tensor[float32, 3], 
                     t_ys: Tensor[int32, 1], t_xs: Tensor[int32, 1]):
    
    b = paths.shape[0]
    max_neg_val = -1e9
    
    # Parallel loop over batch (range(len) or range(b))
    # Note: range(b) works if b is int.
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]
        
        v_prev = 0.0
        v_cur = 0.0
        index = t_x - 1
        
        for y in range(t_y):
            # Explicit cast to int to ensure correct loop bounds
            # My compiler treats min/max as float math functions primarily
            x_start = int(max(0, t_x + y - t_y))
            x_end = int(min(t_x, y + 1))
                
            for x in range(x_start, x_end):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                
                val_max = v_prev
                if v_cur > v_prev:
                    val_max = v_cur
                    
                value[y, x] += val_max

        # Backtrack
        y = t_y - 1
        while y >= 0:
            path[y, index] = 1
            if index != 0:
                should_dec = 0
                if index == y:
                    should_dec = 1
                else:
                    # check values[y-1, index] < values[y-1, index-1]
                    if value[y - 1, index] < value[y - 1, index - 1]:
                        should_dec = 1
                
                if should_dec == 1:
                    index = index - 1
            y -= 1

def test_maximum_path():
    print("=" * 60)
    print("Testing Monotonic Alignment Search (Viterbi)")
    print("=" * 60)
    
    # ----------------------------------------------------
    # Generate Test Data
    # ----------------------------------------------------
    BATCH = 16
    MAX_T = 50
    MAX_S = 200
    
    # Random variables
    t_ys = np.random.randint(10, MAX_T, size=BATCH).astype(np.int32)
    t_xs = np.random.randint(20, MAX_S, size=BATCH).astype(np.int32)
    
    # Values: [B, T, S]
    values = np.random.randn(BATCH, MAX_T, MAX_S).astype(np.float32)
    paths = np.zeros((BATCH, MAX_T, MAX_S), dtype=np.int32)
    
    # ----------------------------------------------------
    # Run CPU Reference
    # ----------------------------------------------------
    print("Running Python/CPU reference...")
    values_cpu = values.copy()
    paths_cpu = paths.copy()
    
    start_cpu = time.perf_counter()
    maximum_path_python(paths_cpu, values_cpu, t_ys, t_xs)
    end_cpu = time.perf_counter()
    print(f"CPU time: {(end_cpu - start_cpu)*1000:.2f} ms")
    
    # ----------------------------------------------------
    # Run GPU (PyCUDA Compiler)
    # ----------------------------------------------------
    print("Running GPU kernel...")
    print("=== Generated CUDA Code ===")
    code = maximum_path_jit.cuda_code
    print(code)
    with open('debug_cuda.cu', 'w') as f:
        f.write(code)
    print("===========================")
    
    # Exit early to inspect code
    # sys.exit(0)
    
    # Prepare Data
    paths_gpu = cp.asarray(paths)
    values_gpu = cp.asarray(values)
    t_ys_gpu = cp.asarray(t_ys)
    t_xs_gpu = cp.asarray(t_xs)
    
    maximum_path_jit(paths_gpu, values_gpu, t_ys_gpu, t_xs_gpu)
    cp.cuda.Stream.null.synchronize()

    # Reset data after warmup because kernel modifies in-place
    paths_gpu = cp.asarray(paths)
    values_gpu = cp.asarray(values)

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    
    maximum_path_jit(paths_gpu, values_gpu, t_ys_gpu, t_xs_gpu)
    
    end_gpu.record()
    end_gpu.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"GPU time: {gpu_time:.2f} ms")
    
    # ----------------------------------------------------
    # Verification
    # ----------------------------------------------------
    paths_gpu_result = cp.asnumpy(paths_gpu)
    values_gpu_result = cp.asnumpy(values_gpu)
    
    # Compare
    diff_paths = np.abs(paths_cpu - paths_gpu_result).sum()
    print(f"Paths difference count: {diff_paths}")
    
    diff_vals = np.max(np.abs(values_cpu - values_gpu_result))
    print(f"Values max diff: {diff_vals}")
    
    if diff_paths == 0 and diff_vals < 1e-4:
        print("✅ SUCCESS: Results match!")
    else:
        print("❌ FAILURE: Results mismatch.")

if __name__ == '__main__':
    test_maximum_path()
