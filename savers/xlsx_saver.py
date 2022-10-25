from .saver_interface import *

from typing import Union
from os import PathLike
import pandas as pd


class XLSXSaver(SaverInterface):
    def __init__(self, file_path: Union[str, bytes, PathLike]):
        self.__clear_file(file_path)
        self.__xsl_writer = pd.ExcelWriter(file_path, mode='a')

    def __clear_file(self, file_path):
        file = pd.ExcelWriter(file_path, mode='w')
        empty_df = pd.DataFrame()
        empty_df.to_excel(file)
        file.close()

    def save_state(self, state: State):
        weights_list = list(state.weights)
        data = weights_list + [state.error]
        df = pd.DataFrame(data)
        df.to_excel(self.__xsl_writer)
        self.__xsl_writer.save()

    def close(self):
        self.__xsl_writer.close()
