import math

pipeline_x = [142, 148, 145, 151, 143, 152, 146, 149, 144, 150,
              147, 151, 145, 148, 146, 152, 144, 149, 147, 150,
              145, 151, 143, 148, 146, 152, 144, 149, 147, 150]


def count_event(pipeline, event_x_condition):
    count = 0
    observations = []
    for element in pipeline:
        if (event_x_condition(element)):
            observations.append(element)
            count += 1
    return count, observations


def diagnostics(d, e, f):
    countD = len(obs_d)
    countE = len(obs_e)
    countF = len(obs_f)
    if (max(count_d, count_e, count_f) == count_d):
        return "Fast", "event D", "data transmission latency is reduced"
    if (max(count_d, count_e, count_f) == count_e):
        return "Normal", "event E", "data transmission latency is normal"
    if (max(count_d, count_e, count_f) == count_f):
        return "Slow", "event F", "data transmission latency is increased"

# Event D (fast latency): x <= 143


def event_d_condition(x): return x <= 143


count_d, obs_d = count_event(pipeline_x, event_d_condition)

# Event E: (fast latency): 143 < x <= 147


def event_e_condition(x): return 143 < x <= 147


count_e, obs_e = count_event(pipeline_x, event_e_condition)

# Event F: (slow latency): x > 147


def event_f_condition(x): return x > 147


count_f, obs_f = count_event(pipeline_x, event_f_condition)

print(f"Event D: count={count_d}, observations={obs_d}")
print(f"Event E: count={count_e}, observations={obs_e}")
print(f"Event F: count={count_f}, observations={obs_f}")

# Verify partition: A and B should be exhaustive and mutually exclusive
assert count_d + count_e + count_f == len(pipeline_x)
print("D, E, and F partition the space correctly.")

behavior, event, situation = diagnostics(obs_d, obs_e, obs_f)
print(f"Diagnostic: Pipeline X exhibits {behavior}. The majority of requests \
    fall into {event}, indicating {situation}.")

