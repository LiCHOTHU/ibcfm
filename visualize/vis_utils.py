import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# 1) Sample the four‐shape mixture in (x,y)
def sample_shapes(n):
    centers = {
        0: np.array([5.0,  7.0]),   # circle
        1: np.array([11.0, 7.0]),   # square
        # 2: np.array([5.0, -7.0]),   # triangle
        # 3: np.array([11.0,-7.0])    # Gaussian cloud
    }
    X = np.zeros((n,2))
    comps = np.random.choice(len(centers.keys()), size=n)
    for i, comp in enumerate(comps):
        c = centers[comp]
        if comp == 0:
            # θ = 2*np.pi*np.random.rand()
            # X[i] = c + [np.cos(θ), np.sin(θ)]

            # wiggly circle:
            θ = 2 * np.pi * np.random.rand()
            r = 1 + 0.6 * np.sin(12 * θ)  # 8-fold ripple
            X[i] = c + [r * np.cos(θ), r * np.sin(θ)]
        elif comp == 1:
            e, t = np.random.choice(4), np.random.uniform(-1,1)
            if e==0:   X[i] = [c[0]+t, c[1]+1]
            elif e==1: X[i] = [c[0]+1, c[1]+t]
            elif e==2: X[i] = [c[0]+t, c[1]-1]
            else:      X[i] = [c[0]-1, c[1]+t]
        elif comp == 2:
            R, h = 1, np.sqrt(3)/2
            v0 = c + [0, 2*h/3]
            v1 = c + [-R/2, -h/3]
            v2 = c + [R/2, -h/3]
            r1, r2 = np.random.rand(), np.random.rand()
            if r1+r2>1: r1, r2 = 1-r1, 1-r2
            X[i] = v0 + r1*(v1-v0) + r2*(v2-v0)
        else:
            X[i] = np.random.multivariate_normal(c, 0.25*np.eye(2))
    return X