import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Load dataset
df = pd.read_csv("Mall_Customers.csv")
# Keep useful features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def initialize_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        points = X[clusters == i]
        if len(points) > 0:
            new_centroids.append(np.mean(points, axis=0))
        else:
            new_centroids.append(X[np.random.randint(len(X))])
    return np.array(new_centroids)
def k_means(X, k=3, max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters



k = 3
centroids, labels = k_means(X_scaled, k=k)
# Assign labels to original DataFrame
df['Cluster'] = labels




plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(k):
    cluster_points = X_scaled[labels == i]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {i}", color=colors[i])
plt.scatter(centroids[:, 1], centroids[:, 2], color='black', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (normalized)')
plt.ylabel('Spending Score (normalized)')
plt.title('K-Means Clustering of Customers')
plt.legend()
plt.grid(True)
plt.show()




# Un-normalize for interpretation
X_unscaled = scaler.inverse_transform(X_scaled)
df[['Age', 'Annual Income', 'Spending Score']] = X_unscaled
# Group summary
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income', 'Spending Score']].mean()
print("\n Cluster Characteristics:")
print(cluster_summary.round(2))






