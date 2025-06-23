import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from vis_utils import sample_shapes

# 2) DCT & IDCT helpers
def to_freq(X):
    return dct(X, type=2, norm='ortho', axis=1)

def to_space(C):
    return idct(C, type=2, norm='ortho', axis=1)

# 3) Generate and transform ground truth
n_samples = 10000
X_gt = sample_shapes(n_samples)    # (10000,2)
C_gt = to_freq(X_gt)               # (10000,2)

# 4) Train flow‐matching in frequency domain
sigma_max, sigma_min = 8.0, 0.01
N_train = 5000
idx     = np.random.choice(n_samples, N_train, replace=False)
x1 = C_gt[idx]                         # clean freq coefs
x0 = np.random.randn(N_train, 2)       # noise freq
t  = np.random.rand(N_train, 1)
x_t = (1-t)*x1 + t*x0                  # bridge
u_t = x0 - x1                          # target field

# RFF regression
D   = 500
rng = np.random.RandomState(0)
W   = rng.randn(D, 3)
b   = rng.uniform(0, 2*np.pi, (D,1))

def rff(Z):
    P = W.dot(Z.T) + b
    return np.sqrt(2/D) * np.cos(P).T

Phi   = rff(np.hstack((x_t, t)))
W_out = np.linalg.solve(Phi.T@Phi + 1e-3*np.eye(D), Phi.T@u_t)

def u_theta(c, tau):
    inp = np.array([c[0], c[1], tau])[None,:]
    return rff(inp).dot(W_out)[0]

# 5) Reverse‐Euler over multiple initial noise points
num_steps = 200
t_sched   = np.linspace(1.0, 0.0, num_steps)
dt        = t_sched[:-1] - t_sched[1:]
num_traj  = 100
initial_xy = rng.randn(num_traj, 2)
C_paths   = []

for init in initial_xy:
    c = to_freq(init[None,:])[0]   # DCT of the noise point
    traj = [c.copy()]
    for i, tau in enumerate(t_sched[:-1]):
        v = u_theta(c, tau)
        c = c - v * dt[i]
        traj.append(c.copy())
    C_paths.append(np.stack(traj))

C_paths = np.stack(C_paths)  # (num_traj, num_steps, 2)

# 6) IDCT back to (x,y)
X_paths = to_space(C_paths.reshape(-1, 2)).reshape(num_traj, num_steps, 2)

# 7) Plot everything
fig, ax = plt.subplots(figsize=(8,8))

# GT sample scatter
ax.scatter(X_gt[:,0], X_gt[:,1], s=3, alpha=0.2, color='black', label='GT samples')

# # Draw outlines of the four shapes once
# θ = np.linspace(0,2*np.pi,200)
# ax.plot(5+np.cos(θ), 7+np.sin(θ), 'r-', label='Circle')
# sq = np.array([11,7])
# ax.plot([sq[0]-1, sq[0]+1, sq[0]+1, sq[0]-1, sq[0]-1],
#         [sq[1]-1, sq[1]-1, sq[1]+1, sq[1]+1, sq[1]-1],
#         'g-', label='Square')
# tri = np.array([5,-7]); h=np.sqrt(3)
# vts = np.array([tri + [0,2*h/3], tri + [-1,-h/3], tri + [1,-h/3]])
# ax.plot(np.append(vts[:,0], vts[0,0]), np.append(vts[:,1], vts[0,1]),
#         'b-', label='Triangle')
# cloud = np.random.multivariate_normal([11,-7], 0.25*np.eye(2), 200)
# ax.scatter(cloud[:,0], cloud[:,1], s=15, color='gray', alpha=0.5, label='Cloud')

# Plot all trajectories, colored by t_sched
norm = plt.Normalize(vmin=0, vmax=1)
cmap = plt.cm.viridis
for traj in X_paths:
    ax.scatter(traj[:,0], traj[:,1],
               c=t_sched, cmap=cmap, norm=norm, s=8)

# Mark initial noise points
ax.scatter(initial_xy[:,0], initial_xy[:,1],
           marker='*', s=100, color='red', label='Starts')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(t_sched)
fig.colorbar(sm, ax=ax, label='t (σ schedule)')

ax.set(aspect='equal',
       title='Flow Matching in Freq→XY from Multiple DCT Noise Inits',
       xlabel='x', ylabel='y')
ax.grid(True)
ax.legend(fontsize='small')
plt.show()
