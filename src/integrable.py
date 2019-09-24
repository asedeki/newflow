from abc import ABC, abstractmethod
import numpy as np


class Goology():
    _instance = None
    g1 = None
    g2 = None
    g3 = None

    def __new__(cls, *vargs, **kwargs):
        if cls._instance is None:
            cls._instance = super(Goology, cls).__new__(cls, *vargs, **kwargs)
        return cls._instance


class Integrable(ABC):
    Neq = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def initpack(self, *vargs) -> np.ndarray:
        pass

    @abstractmethod
    def pack(self, *vargs) -> np.ndarray:
        pass

    @abstractmethod
    def unpack(self, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def rg_equations(self, *vargs, **kwargs):
        pass
