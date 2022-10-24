
import numpy as np
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import tqdm
import random

from typing import List


class ClusteringAlgo:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters

    @property
    def num_clusters(self):
        return self._num_clusters

    @num_clusters.setter
    def num_clusters(self, new):
        self._num_clusters = new

    def cluster(self, vectors) -> List[int]:
        """
        :param vectors: Данные для кластеризации
        :return: Номер кластера для каждого вектора в vectors
        """
        raise NotImplemented

class SpectralClusteringAlgo(ClusteringAlgo):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self._algo = skcluster.SpectralClustering(n_clusters=num_clusters)

    def cluster(self, vectors):
        return self._algo.fit_predict(vectors)
	


class KMeansClusteringAlgo(ClusteringAlgo):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)
        self._algo = skcluster.KMeans(num_clusters)

    def cluster(self, vectors) -> List[int]:
        self._algo.fit(vectors)
        return self._algo.predict(vectors).tolist()


class RandomClusteringAlgo(ClusteringAlgo):
    def __init__(self, num_clusters):
        super().__init__(num_clusters)

    def cluster(self, vectors) -> List[int]:
        result = list()
        for _ in range(len(vectors)):
            result.append(random.randint(0, self.num_clusters - 1))
        return result


def generate_vectors(n, d):
    return np.random.rand(n, d)


def generate_weights(d):
    return np.diag(np.random.dirichlet(np.random.rand(d), size=1)[0])


def error_func(x, y):
    return 1 - skmetrics.adjusted_rand_score(x, y)


def calc_error(vectors, clustering_algo1, clustering_algo2):
    cluster1 = clustering_algo1.cluster(vectors)
    cluster2 = clustering_algo2.cluster(vectors)
    return error_func(cluster1, cluster2)


def train(vectors, clustering_algo1, clustering_algo2, error_threshold=0, use_tqdm=False):
    n, d = vectors.shape
    best_score = calc_error(vectors, clustering_algo1, clustering_algo2)
    best_weights = np.diag(np.ones(d))
    progress = tqdm.tqdm(range(n * n)) if use_tqdm else range(n * n)
    for _ in progress:
        weights = generate_weights(d)
        weighted_vectors = np.dot(vectors, weights)
        score = calc_error(weighted_vectors, clustering_algo1, clustering_algo2)
        if score < best_score:
            best_score = score
            best_weights = weights
        if best_score < error_threshold:
            break
    return np.diagonal(best_weights), best_score

if __name__ == "__main__":
    clustering_algo1 = KMeansClusteringAlgo(8)
    clustering_algo2 = SpectralClusteringAlgo(8)
    vectors = generate_vectors(10, 5)
    weights, error = train(vectors, clustering_algo1, clustering_algo2, use_tqdm=True)
    print()
    print(weights)
    print()
    print(error)
