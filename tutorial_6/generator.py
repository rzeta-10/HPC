import numpy as np

# Matrix size
N = 10_000  

# Generate a 10,000 x 10,000 matrix with random double-precision values
matrix = np.random.rand(N, N).astype(np.float64)

# Save to a text file
np.savetxt("matrix.txt", matrix, fmt="%.6f")

print("Matrix saved successfully as 'matrix.txt'")
