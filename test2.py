from clustering_algos import *
from criterions import *
from input_generators import *
from normalizers import *
from savers import *
from weights_generators import *
from evaluator import Evaluator

import tqdm


class Tester:
    def __init__(self,
                 criterion,
                 weights_generator,
                 num_samples,
                 N, D,
                 normalizer):
        self.__criterion = criterion
        self.__weights_generator = weights_generator
        input_generator = RandomInputGenerator(normalizer, N, D)
        self.__vectors_dataset = self.__generate_dataset(num_samples, input_generator)

    def __generate_dataset(self, num_samples, input_generator):
        result = list()
        for _ in range(num_samples):
            vectors = input_generator.generate()
            result.append(vectors)
        return result

    def test(self, cl_algo_1, cl_algo_2):
        evaluator = self.__get_evaluator(cl_algo_1, cl_algo_2)
        result = list()
        for sample in self.__vectors_dataset:
            best_state = evaluator.evaluate(sample, save_all=False)
            error = best_state.error
            result.append(error)
        return sum(result) / len(result)

    def __get_evaluator(self, cl_algo_1, cl_algo_2):
        evaluator = Evaluator(
            self.__weights_generator,
            cl_algo_1,
            cl_algo_2,
            self.__criterion,
            PlugSaver()
        )
        return evaluator


def test(tester, cl_algo_1_class, cl_algo_2_class, n_clusters_start, n_clusters_end):
    result = list()
    for n_clusters in tqdm.tqdm(range(n_clusters_start, n_clusters_end)):
        cl_algo_1 = cl_algo_1_class(n_clusters)
        cl_algo_2 = cl_algo_2_class(n_clusters)
        error = tester.test(cl_algo_1, cl_algo_2)
        result.append(error)
    return result


INPUT_FILE = "input.csv"

normalizer = MinMaxNormalizer()

N = 100
D = 5

criterion = AdjustedRandIndexCriterion()
weights_generator = CombWeightsGenerator(D, step=0.1)

NUM_SAMPLES = 10

tester = Tester(
    criterion,
    weights_generator,
    NUM_SAMPLES,
    N, D,
    normalizer
)

errors = test(tester, KMeansClusteringAlgo, SpectralClusteringAlgo, 2, D * D + 1)

print(errors)

import matplotlib.pyplot as plt

x = [i for i in range(2, len(errors) + 2)]
y = errors

fig, ax = plt.subplots()
ax.set_title(f"2 - D*D num clusters for random data, {D = }")
ax.plot(x, y)
ax.grid()
ax.set_xlabel("num clusters")
ax.set_ylabel("error")

plt.savefig(f"plots/num_clusters_error_dependence_D_{D}")
