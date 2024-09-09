import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self,max_iter=100,k=10,conv_th=1e-3):
        self.k = k
        self.max_iter = max_iter
        self.conv_th = conv_th
        self.centroids = None

    def fit(self,X):
        n_samples = X.shape[0]
        X = X.values
        nfeatures = X.shape[1]
        self.centroids = X[np.random.choice(n_samples,size=self.k)]
        for i in range(self.max_iter):
            new_centroids = np.empty((self.k, nfeatures))

            distances = self.EuclideanDistances(X)
            labels = np.argmin(distances, axis=1)

            for j in range(self.k):
                inds = np.where(labels == j)[0]
                if len(inds) == 0:
                    new_centroids[j] = self.centroids[j]
                else:
                    cluster_points = X[inds].reshape(-1, nfeatures)
                    new_centroids[j] = np.mean(cluster_points, axis=0)

            if np.max(np.abs(np.array(new_centroids) - (self.centroids))) < self.conv_th:
                print(f"Convergence reached after {i} iterations")
                break

            self.distances = distances
            self.centroids = new_centroids.copy()

    def predict(self,X):
        X = X.values
        self.distances = self.EuclideanDistances(X)
        self.labels = np.argmin(self.distances, axis=1)
        return self.labels

    def getCost(self):
        closest_cluster_dist = np.min(self.distances, axis=1)
        cost = np.sum(closest_cluster_dist ** 2)
        return cost

    def ElbowMethod(self,X,numk=10):
        k_WCSS = []
        for k in range(1,numk+1):
            self.k = k
            self.fit(X)
            labels = self.predict(X)
            cost = self.getCost()
            k_WCSS.append(cost)

        plt.figure(figsize=(8,6))
        plt.plot(range(1,numk+1),k_WCSS,marker='o')
        plt.xticks(range(1,numk+1))
        print('Elbow method executed successfully.')
        plt.show()
        # return k_WCSS

    def EuclideanDistances(self,X):
        distances = np.linalg.norm(X[:,np.newaxis,:]-self.centroids[np.newaxis,:,:],axis=2)
        return distances