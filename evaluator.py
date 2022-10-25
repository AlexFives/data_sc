from weights_generators import WeightsGeneratorInterface
from clustering_algos import ClusteringAlgoInterface
from criterions import CriterionInterface
from savers import SaverInterface
from utils.state import State

import numpy as np
import tqdm


class Evaluator:
    def __init__(self,
                 weights_generator: WeightsGeneratorInterface,
                 clustering_algo_1: ClusteringAlgoInterface,
                 clustering_algo_2: ClusteringAlgoInterface,
                 criterion: CriterionInterface,
                 saver: SaverInterface):
        self.__weights_generator = weights_generator
        self.__clustering_algo_1 = clustering_algo_1
        self.__clustering_algo_2 = clustering_algo_2
        self.__criterion = criterion
        self.__saver = saver

    def evaluate(self, vectors: np.ndarray, use_tqdm: bool = False):
        weights_iterator = self.__weights_generator.generate()
        if weights_iterator is None:
            print("No combinations")
            return
        if use_tqdm:
            weights_iterator = tqdm.tqdm(weights_iterator)
        best_state = State.bad_state()
        for weights in weights_iterator:
            state = self.__get_state(vectors, weights)
            self.__saver.save_state(state)
            if state.error < best_state.error:
                best_state = state
            if state.error > 1:
                print(state.error, state.weights)
        self.__saver.save_state(best_state)
        return best_state

    def __process_weights(self, weights):
        return np.diag(weights)

    def __get_state(self, vectors, weights) -> State:
        processed_weights = self.__process_weights(weights)
        weighted_vectors = np.dot(vectors, processed_weights)
        error = self.__calculate_error(weighted_vectors)
        state = State(weights, error)
        return state

    def __calculate_error(self, weighted_vectors):
        cluster_distribution_1 = self.__clustering_algo_1.cluster(weighted_vectors)
        cluster_distribution_2 = self.__clustering_algo_2.cluster(weighted_vectors)
        error = self.__criterion(cluster_distribution_1, cluster_distribution_2)
        return error
