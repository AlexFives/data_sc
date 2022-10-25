from abc import ABC, abstractmethod

from utils.state import State


class SaverInterface(ABC):
    """
    Сохраняет состояние куда-то.
    """

    @abstractmethod
    def save_state(self, state: State):
        ...

    def close(self):
        """
        Вызывается при завершении работы объекта.
        Например, чтобы закрыть файл для сохранения.
        """
        ...
