from .input_generator_interface import *


class RandomInputGenerator(InputGeneratorInterface):
    def __init__(self, n: int, d: int):
        """
        :param n: Количество векторов
        :param d: Количество измерений каждого вектора
        """
        self.__n = n
        self.__d = d

    def generate(self) -> np.ndarray:
        return np.random.rand(self.__n, self.__d)
