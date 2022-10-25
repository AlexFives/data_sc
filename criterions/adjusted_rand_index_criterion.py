from .criterion_interface import *
from sklearn.metrics import adjusted_rand_score


class AdjustedRandIndexCriterion(CriterionInterface):
    """
    Ошибка рассчитывается по расстоянию rand index.
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    """

    def __call__(self, x: List[int], y: List[int]) -> float:
        return 1. - adjusted_rand_score(x, y)
