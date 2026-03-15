import pandas as pd

df = pd.read_excel("titanic.csv")
# df_sub = df[df['Age'] > 30] 
# print(df_sub)
print(df[10:20])
f = df[df["Sex"] == "Female"]
print(f)