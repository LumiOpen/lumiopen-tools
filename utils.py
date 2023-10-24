import sys

from functools import wraps
from time import time


def timed(f):
    @wraps(f)
    def timed_f(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'{f.__name__}: {end-start:.1f} seconds', file=sys.stderr)
        return result
    return timed_f
