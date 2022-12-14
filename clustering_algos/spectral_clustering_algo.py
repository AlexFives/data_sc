from .clustering_algo_interface import *

from sklearn.cluster import SpectralClustering


class SpectralClusteringAlgo(ClusteringAlgoInterface):
    def __init__(self, num_clusters: int):
        super().__init__(num_clusters)
        self.__algo = SpectralClustering(n_clusters=num_clusters)

    def cluster(self, vectors) -> List[int]:
        return self.__algo.fit_predict(vectors)
