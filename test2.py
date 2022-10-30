from weights_generators import CombWeightsGenerator

gen = CombWeightsGenerator(10, 0.05)

counter = 0
for x in gen.generate():
    counter += 1
    print(x)

print(f"All: {counter}")
