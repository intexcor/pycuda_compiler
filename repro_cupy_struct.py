
import numpy as np
import cupy as cp

def main():
    print("Testing CuPy structured array support...")
    
    particle_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('mass', np.float32)
    ])
    
    particles = np.zeros(10, dtype=particle_dtype)
    print(f"NumPy array dtype: {particles.dtype}")
    
    try:
        _ = cp.asarray(particles)
        print("Success: cp.asarray(particles)")
    except Exception as e:
        print(f"Failed: cp.asarray(particles)\nError: {e}")

    # Workaround attempt 1: View as bytes (numpy) -> copy -> view as struct (cupy -- if supported)
    # Note: CuPy might not support the structured dtype even if we cast it later?
    try:
        # Just copy bytes
        particles_bytes = particles.view(np.uint8)
        _ = cp.asarray(particles_bytes)
        print("Success: cp.asarray(particles.view(np.uint8))")
        
        # Can we cast it back?
        # gpu_particles_2 = gpu_bytes.view(particle_dtype) 
        # But wait, we need to know if cp.dtype supports it.
        # It seems cp.dtype DOES support structured types but maybe asarray has issues?
        pass 
    except Exception as e:
        print(f"Failed workaround 1: {e}")

if __name__ == "__main__":
    main()
