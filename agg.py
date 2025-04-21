import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load dataset
df = pd.read_csv('Wholesale customers data.csv')
# Drop non-numeric columns (if present)
X = df.drop(columns=['Channel', 'Region'], errors='ignore')
# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.cluster import AgglomerativeClustering
# Agglomerative Clustering model
agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
labels_agg = agg.fit_predict(X_scaled)
# Add cluster labels to the dataframe
df['Agg_Cluster'] = labels_agg


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='complete', metric='euclidean'))
plt.title('Dendrogram (Complete Linkage + Euclidean Distance)')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()


from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
labels_agg = agg.fit_predict(X_scaled)
df['Agg_Cluster'] = labels_agg  # Adding labels to original DataFrame
print(df.head())


from sklearn.decomposition import PCA
# Reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap='Accent', s=50)
plt.title('Hierarchical Clustering Visualization (PCA-2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = labels_kmeans
# Compare Agglomerative vs KMeans
comparison = pd.crosstab(df['Agg_Cluster'], df['KMeans_Cluster'])
print(comparison)
