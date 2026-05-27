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


def count_event(pipeline, event):

    if (event == "A"):
        eventA(pipeline)
    elif (event == "B"):
        eventB(pipeline)
    else:
        eventC(pipeline)


def eventC(pipeline):
    for element in pipeline:
        if (147 < element and element <= 149):
            group = +1
            groupCount = +element
    return group, groupCount


def eventA(pipeline):
    for element in pipeline:
        if (element <= 148):
            group = +1
            groupCount = +element
    print(f"For event A, count is:{groupCount}, and observations are {group}")


def eventB(pipeline):
    for element in pipeline:
        if (element > 148):
            group = +1
            groupCount = +element
    return group, groupCount


def eventB(pipeline):
    for element in pipeline:
        if (element > 148):
            group = +1
            groupCount = +element
    return group, groupCount


count_event(pipeline_a, "A")
