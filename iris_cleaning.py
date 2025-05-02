import pandas as pd

df = pd.read_csv(r"C:\Users\Chaitanya\Downloads\archive (2)\iris.csv")
print(df.head())  # This will show the first 5 rows
# Check basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Check for any missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# View basic statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())
# Remove duplicate rows
df = df.drop_duplicates()

# Check the new shape
print("\nShape after removing duplicates:", df.shape)
# Save cleaned data to a new CSV file
df.to_csv("cleaned_iris.csv", index=False)
import matplotlib.pyplot as plt
import seaborn as sns

# Show class distribution (how many samples of each species)
print("\nSpecies Distribution:")
print(df['species'].value_counts())

# Count plot for species
sns.countplot(x='species', data=df)
plt.title("Species Count")
plt.show()

# Pairplot to see relationships between features
sns.pairplot(df, hue='species')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# Heatmap to see correlation between numerical features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
