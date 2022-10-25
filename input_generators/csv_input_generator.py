from .input_generator_interface import *

from typing import Union
from os import PathLike
import csv


class CSVInputGenerator(InputGeneratorInterface):
    def __init__(self, file_path: Union[str, bytes, PathLike]):
        self.__file_path = file_path

    def generate(self) -> np.ndarray:
        result = list()
        with open(self.__file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                vector = list(map(float, row))
                result.append(vector)
        return np.array(result)
