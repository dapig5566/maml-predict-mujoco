import time


def timed_wrpper(func, *args, **kwargs):
    t1 = time.time()
    ret = func(*args, **kwargs)
    t2 = time.time()
    if not isinstance(ret, tuple):
        ret = (ret,)
    return ret + (t2 - t1,)
