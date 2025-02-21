import numpy as np

# Parameters
num_samples = 1000 # Number of samples in the dataset
sequence_length = 100  # Sequence length
embedding_dim = 500  # Embedding dimension
output_file = "dataset_vectors.txt"  # Output file name

# Generate random data
data = np.random.rand(num_samples, sequence_length * embedding_dim)

# Save to file
np.savetxt(output_file, data, fmt="%.6f")

print(f"Synthetic dataset saved to {output_file}")
