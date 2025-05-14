import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler

#Create a sample DataSet
data = {'Feature1': [1.0, 2.0, None, 4.0],
        'Feature2': [5.0, None,  7.0, 8.0]}

df = pd.DataFrame(data)

# Handle missing values
imputer = SimpleImputer(strategy='mean')

df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Normalize the data
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

print('Processed DataFrame: ', df_normalized)


