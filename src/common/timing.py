from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()
    running = True
    elapser = lambda: default_timer() - start if running else end - start
    yield lambda: elapser()
    end = default_timer()
