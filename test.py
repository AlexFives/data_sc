import matplotlib.pyplot as plt
import csv
import numpy as np

all_weights = list()
errors = list()

with open("output.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        error = float(row[0])
        weights = row[2:]
        weights = [float(w) for w in weights]
        all_weights.append(np.array(weights))
        errors.append(error)

from input_generators import CSVInputGenerator
from normalizers import *

normalizer = MaxNormalizer()
input_generator = CSVInputGenerator(normalizer, "input.csv")

vectors = input_generator.generate()

import sys

print(vectors)
# sys.exit(0)

from clustering_algos import KMeansClusteringAlgo, SpectralClusteringAlgo

NUM_CLUSTERS = 4

kmeans = KMeansClusteringAlgo(NUM_CLUSTERS)
spectral = SpectralClusteringAlgo(NUM_CLUSTERS)


def cluster_vectors(vectors):
    kmeans_clusters = kmeans.cluster(vectors)
    spectral_clusters = spectral.cluster(vectors)
    return kmeans_clusters, spectral_clusters


def get_color(cluster):
    if cluster == 1:
        return 'red'
    if cluster == 2:
        return 'blue'
    else:
        return 'green'


def draw_points(axs, clusters, vectors):
    for i, cluster in enumerate(clusters):
        color = get_color(cluster)
        vector = vectors[i]
        x, y = vector
        axs.scatter(x, y, color=color)


def draw_distribution(vectors, i):
    fig, axs = plt.subplots(2)

    kmeans_clusters, spectral_clusters = cluster_vectors(vectors)

    draw_points(axs[0], kmeans_clusters, vectors)
    draw_points(axs[1], spectral_clusters, vectors)

    plt.savefig(f"plots/w{i}.png")
    plt.clf()
    plt.cla()


draw_distribution(vectors, 0)

for i, weights in enumerate(all_weights):
    weights_diag = np.diag(weights)
    new_vectors = np.dot(vectors, weights_diag)
    draw_distribution(new_vectors, i)

# 0.05742552768181741 0.6315423906153572 0.36845760938464267
