import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Load data
df = pd.read_csv("Wholesale customers data.csv")
X = df.drop(columns=["Channel", "Region"], errors='ignore')  # Optional drops
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




from scipy.spatial.distance import euclidean
from collections import defaultdict
def compute_distance_matrix(X):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean(X[i], X[j])
            D[i][j] = D[j][i] = dist
    return D
def complete_linkage(c1, c2, D):
    return max(D[i][j] for i in c1 for j in c2)
def agglomerative_clustering(X):
    D = compute_distance_matrix(X)
    n = len(X)
    clusters = {i: [i] for i in range(n)}
    history = []
    while len(clusters) > 1:
        keys = list(clusters.keys())
        min_dist = float('inf')
        merge_pair = (None, None)

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                c1, c2 = keys[i], keys[j]
                dist = complete_linkage(clusters[c1], clusters[c2], D)
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (c1, c2)

        i, j = merge_pair
        clusters[n] = clusters[i] + clusters[j]
        history.append((i, j, min_dist, len(clusters[n])))

        del clusters[i], clusters[j]
        n += 1
    return history




from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(history):
    import scipy.cluster.hierarchy as sch
    Z = np.array(history)
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Index")
    plt.ylabel("Distance")
    plt.show()
# Build dendrogram
hist = agglomerative_clustering(X_scaled)
plot_dendrogram(hist)




from scipy.cluster.hierarchy import fcluster
Z = np.array(hist)
k = 3
labels_hc = fcluster(Z, k, criterion='maxclust')
df['HC_Cluster'] = labels_hc




from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for c in np.unique(labels_hc):
    plt.scatter(X_pca[labels_hc == c, 0], X_pca[labels_hc == c, 1], label=f"Cluster {c}")
plt.title("Hierarchical Clustering (PCA 2D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.show()





from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = labels_km
# Crosstab to compare clustering results
comparison = pd.crosstab(df['HC_Cluster'], df['KMeans_Cluster'])
print("\n📊 Cluster Comparison:\n", comparison)

