from .clustering_algo_interface import *

from sklearn.cluster import KMeans


class KMeansClusteringAlgo(ClusteringAlgoInterface):
    def __init__(self, num_clusters: int):
        super().__init__(num_clusters)
        self.__algo = KMeans(n_clusters=num_clusters)

    def cluster(self, vectors) -> List[int]:
        self.__algo.fit(vectors)
        return self.__algo.predict(vectors)
