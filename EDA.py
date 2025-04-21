import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

df = pd.read_csv("your_dataset.csv")  # Replace with your file path

print(" Dataset Shape:", df.shape)
print("\n Data Types:\n", df.dtypes)
print("\n First 5 Rows:\n", df.head())

print("\n Missing Values:\n", df.isnull().sum())
# Optionally fill or drop

print("\n Summary Statistics:\n", df.describe())

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Histograms
df[numerical_cols].hist(bins=30, figsize=(14, 8), layout=(len(numerical_cols)//3 + 1, 3))
plt.tight_layout()
plt.suptitle("📊 Feature Distributions", fontsize=16)
plt.show()

# Boxplots
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[numerical_cols])
plt.title(" Boxplots for Outlier Detection")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("🔗 Feature Correlation")
plt.show()

for col in categorical_cols:
    print(f"\n {col} unique values:", df[col].nunique())
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))  # basic encoding

def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

df_no_outliers = remove_outliers_iqr(df[numerical_cols])

# Choose one
scaler = StandardScaler()  # or MinMaxScaler()
X_scaled = scaler.fit_transform(df_no_outliers)
df_scaled = pd.DataFrame(X_scaled, columns=numerical_cols)

print("\n Scaled Data Sample:\n", df_scaled.head())



import pandas as pd
# Sample numerical DataFrame
data = {
    'Age': [23, 45, 31, 52, 38],
    'Income': [45000, 88000, 60000, 120000, 75000]
}
df = pd.DataFrame(data)
# Function for Min-Max Normalization
def min_max_normalize(df):
    df_normalized = df.copy()
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    return df_normalized
# Apply normalizatio
normalized_df = min_max_normalize(df)
# Result
print(" Original:\n", df)
print("\n Normalized:\n", normalized_df)

# Function for Standardization
def z_score_standardize(df):
    df_standardized = df.copy()
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        df_standardized[column] = (df[column] - mean) / std
    return df_standardized
standardized_df = z_score_standardize(df)
# Result
print("\n Z-Score Standardized:\n", standardized_df)
