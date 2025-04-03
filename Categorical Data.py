
import pandas as pd
data = {'genre': ['action', 'comedy', 'drama', 'horror', 'thriller', 'comedy']} 
df =pd.DataFrame(data)

one_hot_encoded = pd.get_dummies(df['genre'], prefix='genre')

print(f"orginal Data: {df}")
print(f"One-hot Encoded: {one_hot_encoded}")
