from abc import ABC, abstractmethod
import numpy as np

from normalizers import NormalizerInterface


class InputGeneratorInterface(ABC):
    def __init__(self, normalizer: NormalizerInterface):
        self._normalizer = normalizer

    @abstractmethod
    def generate(self) -> np.ndarray:
        ...
