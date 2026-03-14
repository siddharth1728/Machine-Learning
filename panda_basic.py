import pandas as pd
df = pd.read_excel("titanic.csv.xlsx")
a = df.describe()
print(df.head(11)) #it list 5 records

# print(df)   
# print(type(df))
# print(df.dtypes)
# print(df.shape)
# print(a)