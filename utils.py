import functools


def product(ps): return functools.reduce(lambda x, y: x * y, ps, 1)
