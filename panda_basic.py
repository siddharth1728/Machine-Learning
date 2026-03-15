import pandas as pd

# Load the dataset
df = pd.read_csv("titanic.csv")

# ── STEP 1: Explore the data first ──────────────────────────────────────────

print("Shape of dataset:")
print(df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())  # Always print the result so it actually displays

# ── STEP 2: Fill missing values with the mean ────────────────────────────────

# Age is fairly symmetric, so mean is appropriate here.
# We use direct assignment (df['col'] = ...) instead of inplace=True
# to avoid the FutureWarning from pandas 3.0
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fare has outliers (some passengers paid huge amounts), so median
# is safer here — it won't get pulled by extreme values
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# For categorical (text) columns, we use the mode (most frequent value)
# instead of mean, because you can't average text
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ── STEP 3: Remove duplicates ────────────────────────────────────────────────

# This is called directly on df (not on a column slice),
# so inplace=True is safe here — no FutureWarning
df.drop_duplicates(inplace=True)

# ── STEP 4: Verify everything looks good ────────────────────────────────────

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())  # Every column should now show 0

print("\nCleaned dataset (first 10 rows):")
print(df.head(10))

print("\nBasic statistics of the cleaned data:")
print(df.describe())  # Gives min, max, mean, std for all numerical columns