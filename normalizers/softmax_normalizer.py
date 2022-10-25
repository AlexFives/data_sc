from .normalizer_interface import *


class SoftmaxNormalizer(NormalizerInterface):
    def normalize(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / sum(exp)
