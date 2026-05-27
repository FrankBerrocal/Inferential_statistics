import math

pipeline_a = [145, 148, 147, 149, 146, 150,
              148, 147, 149, 146, 148, 147, 149, 150, 148]
x = 0
# if (147 < x and x <= 149):
#     print("Group 2: {x}")
# elif (x > 149):
#     print("Group 3: {x}")
# else:
#     print("Group 1: {x}")

pipeline_a = [145, 148, 147, 149, 146, 150,
              148, 147, 149, 146, 148, 147, 149, 150, 148]


def count_event(pipeline, event_x_condition):
    count = 0
    observations = []
    for element in pipeline:
        if(event_x_condition(element)):
            observations.append(element)
            count += 1
    return count, observations


# Event A: x ≤ 148
def event_a_condition(x): return x <= 148 #returns true or false
count_a, obs_a = count_event(pipeline_a, event_a_condition)

# Event B: x > 148
def event_b_condition(x): return x > 148
count_b, obs_b = count_event(pipeline_a, event_b_condition)

# Event C: 147 < x ≤ 149
def event_c_condition(x): return 147 < x <= 149
count_c, obs_c = count_event(pipeline_a, event_c_condition)

print(f"Event A: count={count_a}, observations={obs_a}")
print(f"Event B: count={count_b}, observations={obs_b}")
print(f"Event C: count={count_c}, observations={obs_c}")

# Verify partition: A and B should be exhaustive and mutually exclusive
assert count_a + count_b == len(pipeline_a)
print("A and B partition the space correctly.")
