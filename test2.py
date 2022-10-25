from weights_generators import CycleWeightsGenerator

D = 4

weights_generator = CycleWeightsGenerator(D, step=0.05)

for weights in weights_generator.generate():
    print(weights)
