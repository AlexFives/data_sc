from .normalizer_interface import *
from pymcdm.normalizations import max_normalization


class MaxNormalizer(NormalizerInterface):
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return max_normalization(x)
