from evaluator import Evaluator
from weights_generators import DirichletWeightsGenerator
from clustering_algos import KMeansClusteringAlgo, SpectalClusteringAlgo
from criterions import AdjustedRandIndexCriterion
from savers import CSVSaver
from input_generators import RandomInputGenerator, CSVInputGenerator

NUM_ITERATIONS = 1000
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

# input_generator = CSVInputGenerator(INPUT_FILE)
input_generator = RandomInputGenerator(20, 5)

vectors = input_generator.generate()

N, D = vectors.shape

weights_generator = DirichletWeightsGenerator(D, NUM_ITERATIONS)

clustering_algo_1 = KMeansClusteringAlgo(D)
clustering_algo_2 = SpectalClusteringAlgo(D)

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
