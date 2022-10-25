from .saver_interface import *

from typing import Union
from os import PathLike
import csv


class CSVSaver(SaverInterface):
    def __init__(self, file_path: Union[str, bytes, PathLike]):
        self.__file = open(file_path, 'w')
        self.__writer = csv.writer(self.__file)

    def save_state(self, state: State):
        """
        Формат сохранения:
        A           B   C...
        state.error ' ' state.weights
        """
        row = [state.error, ' '] + list(state.weights)
        self.__writer.writerow(row)

    def close(self):
        self.__file.close()
