from .input_generator_interface import *


class RandomInputGenerator(InputGeneratorInterface):
    def __init__(self, normalizer: NormalizerInterface,
                 n: int, d: int):
        """
        :param n: Количество векторов
        :param d: Количество измерений каждого вектора
        """
        super().__init__(normalizer)
        self.__n = n
        self.__d = d

    def generate(self) -> np.ndarray:
        weights = np.random.rand(self.__n, self.__d)
        normalized_weights = self._normalizer.normalize(weights)
        return normalized_weights
