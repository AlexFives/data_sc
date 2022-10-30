from .weights_generator_interface import *

import itertools
import math


class CombWeightsGenerator(WeightsGeneratorInterface):
    def __init__(self, d: int, step: float):
        super().__init__(d)
        self.__step = step
        # self.__p = abs(math.floor(math.log10(step)))
        # self.__simple_numbers = self.__generate_d_simple_numbers(d)

    # def __generate_d_simple_numbers(self, d):
    #     result = [2]
    #     i = 3
    #     while len(result) < d:
    #         if self.__is_simple(i):
    #             result.append(i)
    #         i += 2
    #     return result
    #
    # def __is_simple(self, number):
    #     if number in (2, 3, 5, 7):
    #         return True
    #     for i in range(3, math.ceil(math.sqrt(number)), 2):
    #         if number % i == 0:
    #             return False
    #     return True

    def generate(self) -> Iterator[np.ndarray]:
        vector = [1 / self._d] * self._d
        if self.__step >= 1 / self._d:
            yield np.array(vector)
            return
        yield np.array(vector)
        hashes = set()
        for vec in self.__generate(vector):
            # vec_hash = self.__hash_vector(vec)
            vec_hash = hash(tuple(vec))
            if vec_hash not in hashes:
                hashes.add(vec_hash)
                yield np.array(vec)

    def __generate(self, vector):
        if len(vector) == 1:
            yield vector
            return
        while abs(vector[0] - self.__step) > self.__step / 2:
            vector[0] -= self.__step
            vector[1] += self.__step
            yield vector
            for permutation in self.__permutations(vector.copy()):
                yield permutation
            for other_combination in self.__generate(vector[1:]):
                yield [vector[0]] + list(other_combination)

    def __permutations(self, vector):
        return itertools.permutations(vector)

    # def __hash_vector(self, vector):
    #     vector_hash = 0
    #     for x, simple_number in zip(vector, self.__simple_numbers):
    #         vector_hash += x * simple_number
    #     vector_hash = round(vector_hash, self.__p)
    #     return hash(vector_hash)
