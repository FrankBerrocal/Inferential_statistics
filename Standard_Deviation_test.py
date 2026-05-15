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
test_1 = [2, 4, 6, 8, 10]
mean_1, std_1 = analyze_latency(test_1)
assert abs(mean_1 - 6.0) < 1e-9
assert abs(std_1 - 2.828) < 0.01
print("Test 1 passed")

# Test 2
test_2 = [5, 5, 5, 5, 5]
mean_2, std_2 = analyze_latency(test_2)
assert abs(mean_2 - 5.0) < 1e-9
assert abs(std_2 - 0.0) < 1e-9
print("Test 2 passed")

# Test 3
test_3 = [148, 152, 147, 151, 149, 150, 148, 151, 147, 152]
mean_3, std_3 = analyze_latency(test_3)
assert abs(mean_3 - 149.5) < 1e-9
assert abs(std_3 - 1.88) < 0.05
print("Test 3 passed")

print("All tests passed.")
