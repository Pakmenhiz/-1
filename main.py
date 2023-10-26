import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth

# Вхідні дані
data = np.loadtxt('lad01.txt', delimiter=',')

# Крок 1: Визначення кількості кластерів методом зсуву середнього
bandwidth = estimate_bandwidth(data, quantile=0.15, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True )
ms.fit(data)
labels = ms.labels_
n_clusters_ = len(np.unique(labels))

# Крок 2: Оцінка кількості кластерів методом silhouette_score
scores = []
for i in range(2, 16):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    score = silhouette_score(data, labels)
    scores.append(score)

# Крок 3: Вибір оптимальної кількості кластерів
optimal_clusters = np.argmax(scores) + 2  # +2 бо рахуємо з 2 кластерів

# Крок 4: Кластеризація методом k-середніх з оптимальною кількістю кластерів
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_

# Вивід результатів
# Рисунок 1: Вихідні точки на площині
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], marker='o', facecolors='none',
        edgecolors='black', s=80)
plt.title('Вихідні точки')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

# Рисунок 2: Центри кластерів (метод зсуву середнього)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], s=100, c='red')
plt.title('Центри кластерів (Метод зсуву середнього)')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

# Рисунок 3: Бар діаграмма score(number of clusters)
plt.figure(figsize=(8, 6))
plt.bar(range(2, 16), scores)
plt.title('Бар діаграма оцінки (silhouette score)')
plt.xlabel('Кількість кластерів')
plt.ylabel('Оцінка (silhouette score)')

# Рисунок 4: Кластеризовані дані з областями кластеризації
h = .02  # Для визначення шагу сітки в графіку
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
plt.scatter(data[:,0], data[:,1],
        marker='o', s=110, linewidths=4, color='black',
        zorder=12, facecolors='black')
plt.title('Кластеризовані дані з областями кластеризації')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

plt.show()