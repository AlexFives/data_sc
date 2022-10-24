from abc import ABC, abstractmethod
from typing import List


class ClusteringAlgoInterface(ABC):
    def __init__(self, num_clusters: int):
        self._num_clusters = num_clusters

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    @abstractmethod
    def cluster(self, vectors) -> List[int]:
        ...
