from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


class WeightsGeneratorInterface(ABC):
    def __init__(self, d: int):
        """
        :param d: Количество измерений в векторе весов.
        """
        self._d = d

    @abstractmethod
    def generate(self) -> Iterator[np.ndarray]:
        """
        :return: Генерирует векторы весов с количеством измерений == self._d.
                 Веса в сумме дают 1.
        """
        ...
