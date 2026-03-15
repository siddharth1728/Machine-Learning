import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading your dataset
df = pd.read_excel("titanic.csv.xlsx")  # Fix 1: use read_excel() for .xlsx files

# Checking for missing values in each column
# missing_values = df.isnull().sum()
# print(missing_values)

# Visualizing outliers using a box plot
sns.boxplot(x=df['Age'])  # Replace 'your_column' with your column of interest
plt.show()  # Fix 2: removed the erroneous space between plt. and show()