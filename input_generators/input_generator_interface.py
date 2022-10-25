from abc import ABC, abstractmethod
import numpy as np


class InputGeneratorInterface(ABC):
    @abstractmethod
    def generate(self) -> np.ndarray:
        ...
