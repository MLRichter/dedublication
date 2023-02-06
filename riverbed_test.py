import pandas as pd

df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, index=[0, 1, 2])
df2 = pd.DataFrame({'col1': [7, 8, 9], 'col2': [10, 11, 12]}, index=[1, 2, 3])

filtered_df = df1[df1.index.isin(df2.index)]
print(filtered_df)
