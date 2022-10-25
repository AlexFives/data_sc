from .input_generator_interface import *

from typing import Union
from os import PathLike
import csv
import pymcdm


class CSVInputGenerator(InputGeneratorInterface):
    def __init__(self, normalizer: NormalizerInterface,
                 file_path: Union[str, bytes, PathLike]):
        super().__init__(normalizer)
        self.__file_path = file_path

    def generate(self) -> np.ndarray:
        result = list()
        with open(self.__file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                row = [item.replace(',', '.') for item in row]
                vector = list(map(float, row))
                result.append(vector)
        weights = np.array(result)
        normalized_weights = self._normalizer.normalize(weights)
        print(normalized_weights)
        return normalized_weights
