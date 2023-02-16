#step function f(x)=x for x \in [0,10]
import numpy as np


def f_l(hl,x):
    def fx(a):
        return a
    # number of steps
    u=10. #0<=x<=u
    sl = u / hl
    if sl%1>0:
        raise ValueError("number of steps is not a integer!")
    # the interval contain x x1<=x<x2 ; i \in [0,sl-1]
    si = np.floor(x / hl)
    si[x == u] = si[x == u]-1
    x1 = hl * si
    x2 = hl * (si + 1)
    y = fx((x1 + x2) / 2)
    return y












