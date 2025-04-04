from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
data_points = np.random.randn(100, 2) * 2 + np.array([5,6])
kmeans =KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_points)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(f"Cluster labels: {labels}")
print(f"Centroids: {centroids}")



plt.scatter(data_points[:,0], data_points[:,1], c=labels, cmap='viridis', s=50, alpha=0.7)


plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()
