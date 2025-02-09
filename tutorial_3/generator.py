import numpy as np

def generate_doubles_np(n, lower=0, upper=1e6):
    return np.random.uniform(lower, upper, int(n))  # Convert n to integer

# Generate 1 million random double numbers
numbers = generate_doubles_np(1e6)

# Save to a text file
np.savetxt("random_doubles_numpy.txt", numbers, fmt="%.6f")

print("Random double numbers saved to random_doubles_numpy.txt")
