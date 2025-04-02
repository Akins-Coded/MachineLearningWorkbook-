import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(0, 1, 100)

plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Data Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

mean_value = np.mean(data)
median_value = np.median(data)
std_deviation = np.std(data)

print(f"Mean: {mean_value}")
print(f"Median: {median_value} ")
print(f"Standard Deviation: {std_deviation}")