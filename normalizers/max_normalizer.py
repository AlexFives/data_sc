from .normalizer_interface import *
# from pymcdm.normalizations import max_normalization
from pymcdm.normalizations import *


class MaxNormalizer(NormalizerInterface):
    def normalize(self, x: np.ndarray) -> np.ndarray:
        # return max_normalization(x)
        return minmax_normalization(x)
