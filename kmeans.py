import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
# Drop non-numeric columns (e.g., CustomerID, Gender if needed)
data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
# Normalize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)



from sklearn.cluster import KMeans
# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
# Assign cluster labels
df['Cluster'] = kmeans.labels_



import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(
        df[df['Cluster'] == i]['Annual Income (k$)'],
        df[df['Cluster'] == i]['Spending Score (1-100)'],
        s=60,
        c=colors[i],
        label=f'Cluster {i}'
    )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
plt.grid(True)
plt.show()


df['Cluster'] = kmeans.labels_
print(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head())


cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)
