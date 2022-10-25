from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    weights: np.ndarray
    error: float

    @classmethod
    def bad_state(cls):
        return cls(np.ndarray([0]), 10.0)
