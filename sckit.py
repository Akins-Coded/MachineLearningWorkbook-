import numpy as np
import pandas as pd

array = np.array([1, 2, 3, 4, 5])
print("NumPy Array: ", array)
print("Mean of Array: ", np.mean(array))

#pandas operations

data_set = {'NAme': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data_set)
print("Pandas DataFrame: ", df)
print("Average Age: ", df['Age'].mean())

# Operations with file handling

data = {'Name': ['John', 'Jane', 'Doe'],'Score': [85, 90, 78]}
df = pd.DataFrame(data)

# Save DataFrame to CSV file
df.to_csv('data.csv', index=False)
# Load DataFrame from CSV file
loaded_df = pd.read_csv('data.csv')
print("Loaded DataFrame: ", loaded_df)