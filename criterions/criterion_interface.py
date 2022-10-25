from abc import ABC, abstractmethod
from typing import List


class CriterionInterface(ABC):
    @abstractmethod
    def __call__(self, x: List[int], y: List[int]) -> float:
        """
        :param x: Разбиение по кластерам алгоритма №1.
        :param y: Разбиение по кластерам алгоритма №2.
        :return: Ошибка между двумя алгоритмами. Чем ближе к 0, тем меньше ошибка.
        """
        ...
