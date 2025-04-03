from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

linkage_matrix = linkage(X, method='average')

plt.figure(figsize=(12,8))
dendrogram(linkage_matrix)

plt.title('Hierarchical Dendrogram')
plt.xlabel('data point')
plt.ylabel('Distance')
plt.show()