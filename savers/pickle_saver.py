from .saver_interface import *

import pickle
from os import PathLike
from typing import Union


class PickleSaver(SaverInterface):
    def __init__(self, file_path: Union[str, bytes, PathLike]):
        self.__file = open(file_path, 'wb')

    def save_state(self, state: State):
        state_tuple = (state.weights, state.error)
        pickle.dump(state_tuple, self.__file)

    def close(self):
        self.__file.close()
