
import numpy as np
import cupy as cp

try:
    particle_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('mass', np.float32)
    ])

    print("Creating numpy array...")
    particles_cpu = np.zeros(10, dtype=particle_dtype)
    print("Numpy array created. Dtype:", particles_cpu.dtype)

    print("Attempting cp.asarray...")
    particles_gpu = cp.asarray(particles_cpu)
    print("Success!")
except Exception as e:
    print("Caught exception:")
    print(e)
