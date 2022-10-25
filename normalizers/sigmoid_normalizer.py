from .normalizer_interface import *


class SigmoidNormalizer(NormalizerInterface):
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(x))
