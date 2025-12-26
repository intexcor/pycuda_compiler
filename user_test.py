
import sys
import os

# Support running from the current directory
sys.path.append(os.getcwd())

try:
    from compiler import cuda_compile, Array, float32
except ImportError:
    # Fallback to local import
    from compiler import cuda_compile, Array, float32

@cuda_compile
def process(data: Array[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = sqrt(data[i])
        else:
            data[i] = data[i] * data[i]

def main():
    print("Initializing data...")
    data = process.array([4.0, -2.0, 9.0, -3.0])
    
    print("Running kernel...")
    process(data)
    
    print("Fetching results...")
    result = process.to_numpy(data)
    print(f"Result: {result}")
    
    expected = [2.0, 4.0, 3.0, 9.0]
    # Check approximately (floats)
    import numpy as np
    if np.allclose(result, expected):
        print("✅ Success! Matches expected output.")
    else:
        print(f"❌ Failed. Expected {expected}")

if __name__ == '__main__':
    main()
