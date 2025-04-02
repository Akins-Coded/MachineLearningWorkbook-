import numpy

data = [12, 18, 14, 20, 16]
mean = sum(data) /len(data)
print(f"Mean: {mean}")

sorted_data = sorted(data)
n = len(sorted_data)

median = (sorted_data[n // 2] + sorted_data[(n - 1) // 2]) / 2
print(f"Median: {median}")

from statistics import mode
mode_value = mode(data)
print(f"Mode: {mode_value}")