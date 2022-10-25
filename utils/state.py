from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    weights: np.ndarray
    error: float

    @classmethod
    def bad_state(cls):
        return cls(None, 10.0)
