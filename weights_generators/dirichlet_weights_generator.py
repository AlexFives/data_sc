from .weights_generator_interface import *


class DirichletWeightsGenerator(WeightsGeneratorInterface):
    """
    Работает на основе распределение Дирихрета.
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html
    """

    def __init__(self, d: int, num_iterations: int):
        """
        :param d: Количество измерений в векторе весов.
        :param num_iterations: Сколько раз генерировать веса.
        """
        super().__init__(d)
        self.__num_iterations = num_iterations

    def generate(self) -> Iterator[np.ndarray]:
        for _ in range(self.__num_iterations):
            yield np.random.dirichlet(np.ones(self._d), size=1)[0]
