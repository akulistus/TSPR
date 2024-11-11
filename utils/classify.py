import numpy as np

def classify(func, tuple: tuple) -> int:
    if (func(tuple[0], tuple[1]) > 0):
        return 0
    else:
        return 1