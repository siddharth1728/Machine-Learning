import pandas as pd

df = pd.read_excel("titanic.csv.xlsx")

m = df.groupby("Age").mean(numeric_only=True)
print(m)

# print(m)
# print(df)   
# print(type(df))
# print(df.dtypes)
# print(df.columns)
# print(df.axes)
# print(df.Sex)
# print()
# print(df.shape)
# print(a)
