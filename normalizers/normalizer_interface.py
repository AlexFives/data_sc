from abc import ABC, abstractmethod
import numpy as np


class NormalizerInterface(ABC):
    @abstractmethod
    def normalize(self, x: np.ndarray) -> np.ndarray:
        ...
