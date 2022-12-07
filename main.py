from evaluator import Evaluator
from weights_generators import *
from normalizers import *
from clustering_algos import KMeansClusteringAlgo, SpectralClusteringAlgo
from criterions import AdjustedRandIndexCriterion
from savers import PlugSaver
from input_generators import RandomInputGenerator, CSVInputGenerator

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

normalizer = MinMaxNormalizer()
# normalizer = MaxNormalizer()

# input_generator = CSVInputGenerator(normalizer, INPUT_FILE)
input_generator = RandomInputGenerator(normalizer, 100, 2)

vectors = input_generator.generate()

print(vectors)

# import numpy as np
# print(np.sum(vectors.transpose(), axis=1))
# import sys
# sys.exit(0)

N, D = vectors.shape
# NUM_CLUSTERS = 10

# NUM_ITERATIONS = 15
# weights_generator = DirichletWeightsGenerator(D, NUM_ITERATIONS)
weights_generator = CombWeightsGenerator(D, step=0.01)

# clustering_algo_1 = KMeansClusteringAlgo(NUM_CLUSTERS)
# clustering_algo_2 = SpectalClusteringAlgo(NUM_CLUSTERS)
#
criterion = AdjustedRandIndexCriterion()


#
# saver = CSVSaver(OUTPUT_FILE)
#
# evaluator = Evaluator(
#     weights_generator,
#     clustering_algo_1,
#     clustering_algo_2,
#     criterion,
#     saver
# )
#
# best = evaluator.evaluate(vectors, use_tqdm=True)
# print(best.error, best.weights)
#
# saver.close()
#
# SAVER = PlugSaver()


def get_best_weights(input, weights_generator, cl_alg1, cl_alg2, criterion):
    evaluator = Evaluator(
        weights_generator,
        cl_alg1,
        cl_alg2,
        criterion,
        PlugSaver()
    )
    best = evaluator.evaluate(input, use_tqdm=True)
    return best


cl_alg1_class = KMeansClusteringAlgo
cl_alg2_class = SpectralClusteringAlgo

deltas = list()

prev = get_best_weights(
    vectors,
    weights_generator,
    cl_alg1_class(1),
    cl_alg2_class(1),
    criterion,
).weights

for i in range(2, 11):
    best = get_best_weights(
        vectors,
        weights_generator,
        cl_alg1_class(i),
        cl_alg2_class(i),
        criterion,
    ).weights
    new_point = max([abs(a - b) for a, b in zip(best, prev)])
    deltas.append(new_point)
    prev = best

print(deltas)

# [0.3000000000000002, 0.2900000000000002, 0.30000000000000016, 0.29000000000000015, 0.30000000000000016, 0.08999999999999998, 0.1300000000000001, 0.020000000000000018, 0.15000000000000013]

