import functools

product = lambda ps: functools.reduce(lambda x, y: x * y, ps, 1)
