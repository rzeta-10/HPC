import numpy as np
import matplotlib.pyplot as plt

# Load data from performance_results.txt
data = np.loadtxt("performance_results_reduction.txt", delimiter=",", skiprows=1)

# Columns: 1st = Threads, 2nd = Execution Time
threads = data[:, 0].astype(int)  # Convert to integer for clarity
times = data[:, 1]

# Save data into a dictionary {threads -> execution time}
execution_time_dict = dict(zip(threads, times))

# Extract values
threads_list = list(execution_time_dict.keys())
execution_times = list(execution_time_dict.values())

# Plot 1: Execution Time vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads_list, execution_times, color='r', marker='o', linestyle='-', label="Execution Time")
plt.title("Execution Time vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig("execution_time_vs_threads.png")
plt.show()

# Compute Speedup: Speedup = T1 / Tp
speedup = [execution_times[0] / execution_times[i] for i in range(1, len(threads_list))]

# Plot 2: Speedup vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads_list[1:], speedup, color='b', marker='o', linestyle='-', label="Measured Speedup")
plt.title("Speedup vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.legend()
plt.grid(True)

# Adjust y-axis to whole numbers
max_speedup = int(np.ceil(max(speedup)))
plt.yticks(range(0, max_speedup + 1))
plt.ylim(0, max_speedup)

plt.savefig("speedup_vs_threads.png")
plt.show()

# Compute Parallelization Factor using Amdahl's Law
parallel_factors = [
    ((1 / speedup[i - 1]) - 1) / ((1 / threads_list[i]) - 1) for i in range(1, len(threads_list))
]

# Plot 3: Parallelization Factor vs Threads
plt.figure(figsize=(8, 6))
plt.plot(threads_list[1:], parallel_factors, color='g', marker='o', linestyle='-', label="Parallelization Factor")
plt.title("Parallelization Factor vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Parallelization Factor")
plt.legend()
plt.grid(True)
plt.savefig("parallel_fraction_vs_threads.png")
plt.show()

# Compute and display the average parallelization fraction
if parallel_factors:
    avg_parallel_fraction = np.mean(parallel_factors)
    print(f"Estimated Parallelization Fraction: {avg_parallel_fraction:.4f}")
    max_parallel_fraction = max(parallel_factors)
    print(f"Maximum Parallelization Fraction: {max_parallel_fraction:.4f}")
    print(f"The threads with the maximum parallelization fraction is {threads_list[parallel_factors.index(max_parallel_fraction) + 1]}")
else:
    print("Error: Could not compute parallelization fraction due to invalid speedup values.")
