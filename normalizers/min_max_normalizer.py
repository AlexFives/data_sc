from .normalizer_interface import *

from sklearn.preprocessing import MinMaxScaler


class MinMaxNormalizer(NormalizerInterface):
    def __init__(self):
        self.__scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self.__scaler.fit_transform(x)
