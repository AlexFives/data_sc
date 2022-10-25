from evaluator import Evaluator
from weights_generators import DirichletWeightsGenerator, CycleWeightsGenerator
from normalizers import *
from clustering_algos import KMeansClusteringAlgo, SpectalClusteringAlgo
from criterions import AdjustedRandIndexCriterion
from savers import CSVSaver
from input_generators import RandomInputGenerator, CSVInputGenerator

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

normalizer = MinMaxNormalizer()

input_generator = CSVInputGenerator(normalizer, INPUT_FILE)
# input_generator = RandomInputGenerator(normalizer, 100, 2)

vectors = input_generator.generate()

N, D = vectors.shape
NUM_CLUSTERS = 3

# NUM_ITERATIONS = 1000
# weights_generator = DirichletWeightsGenerator(D, NUM_ITERATIONS)
weights_generator = CycleWeightsGenerator(D, step=0.1)

clustering_algo_1 = KMeansClusteringAlgo(NUM_CLUSTERS)
clustering_algo_2 = SpectalClusteringAlgo(NUM_CLUSTERS)

criterion = AdjustedRandIndexCriterion()

saver = CSVSaver(OUTPUT_FILE)

evaluator = Evaluator(
    weights_generator,
    clustering_algo_1,
    clustering_algo_2,
    criterion,
    saver
)

evaluator.evaluate(vectors, use_tqdm=True)
