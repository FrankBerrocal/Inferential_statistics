import math

import numpy as np
# restrictions: no usage of statistics functions.
# refactor:  one function only
# case 1:  [2, 4, 6, 8, 10]
# case 2:  [5, 5, 5, 5, 5]
# case 3:  [148, 152, 147, 151, 149, 150, 148, 151, 147, 152]


def analyze_latency(random_data):
    mean_1 = sum(random_data)/len(random_data)
    addition = 0
    total = 0
    square = 0
    for element in random_data:
        addition = element - mean_1
        square = math.pow(addition, 2)
        total += square
    std_1 = (math.sqrt(total / len(random_data)))
    return mean_1, std_1


# Test 1
pipeline_a = [145, 148, 147, 149, 146, 150, 148,
              147, 149, 146, 148, 147, 149, 150, 148]
mean_1, std_1 = analyze_latency(pipeline_a)


# Test 2
pipeline_b = [120, 180, 130, 170, 125, 175, 135,
              165, 140, 160, 145, 155, 150, 125, 175]
mean_2, std_2 = analyze_latency(pipeline_b)


print(f"Pipeline A: mean={mean_1}ms, std={std_1}ms ")
print(f"Pipeline B: mean={mean_2}ms, std={std_2}ms ")

print(f"Diagnostics:  Pipeline A is more reliable since dispersion of the observations around the mean is lower")
print("Means have a similar result due to the proximity of the observations in both sets")
print("STD is revealing that the mean of the latency is closed to the mean in pipeline A")
print("Pipeline A is operationally better because the proximity of the latency observations is closer to the mean.")