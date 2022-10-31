from .saver_interface import *


class PlugSaver(SaverInterface):
    def save_state(self, state: State):
        ...

    def close(self):
        ...
