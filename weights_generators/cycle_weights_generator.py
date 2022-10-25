from .weights_generator_interface import *


class CycleWeightsGenerator(WeightsGeneratorInterface):
    def __init__(self, d: int, step: float = 0.1):
        """
        :param d: Количество изменений в векторе весов.
        :param step: Шаг итерации.
                     Т.е. будут перебираться значения от 0 до (1 / step)^d
        """
        super().__init__(d)
        self.__step = step

    def generate(self) -> Iterator[np.ndarray]:
        max_number = int(1 / self.__step) ** self._d
        for i in range(max_number):
            weights = self.__generate_weights(i)
            if not self.__checksum(weights):
                continue
            yield np.array(weights)

    def __generate_weights(self, number):
        weights = list()
        for _ in range(self._d):
            number, reminder = divmod(number, 10)
            weight = reminder * self.__step
            weights.append(weight)
        return weights

    def __checksum(self, weights):
        if 0 in weights:
            return False
        weights_sum = sum(weights)
        return abs(1 - weights_sum) < self.__step
