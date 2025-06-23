import numpy as np
import matplotlib.pyplot as plt
from vis_utils import sample_shapes

# 2. Generate training data
N_data = 10000
x1     = sample_shapes(N_data)
x0     = np.random.randn(N_data, 2)
t_data = np.random.rand(N_data, 1)
xt     = (1 - t_data) * x1 + t_data * x0
ut     = x0 - x1

# 3. Train Random Fourier Feature regressor for u_hat
D   = 500
rng = np.random.RandomState(0)
W   = rng.randn(D, 3)
b   = rng.uniform(0, 2*np.pi, size=(D, 1))

def rff_features(x):
    proj = W.dot(x.T) + b
    return np.sqrt(2/D) * np.cos(proj).T

Phi   = rff_features(np.hstack((xt, t_data)))
reg   = 1e-3
W_out = np.linalg.solve(Phi.T.dot(Phi) + reg*np.eye(D), Phi.T.dot(ut))

def u_hat(x, t):
    x = np.atleast_2d(x)
    inp = np.hstack((x, np.full((len(x),1), t)))
    return rff_features(inp).dot(W_out)

# 4. Simulate reverse ODE from t=1 → 0 for multiple noise samples
num_steps      = 5000
num_pts        = 100
t_rev          = np.linspace(1, 0, num_steps)  # σ schedule
dt_rev         = t_rev[:-1] - t_rev[1:]
initial_points = np.random.randn(num_pts, 2)
trajectories   = [[p.copy()] for p in initial_points]

for i, σ in enumerate(t_rev[:-1]):
    for traj in trajectories:
        x_curr = traj[-1]
        v      = u_hat(x_curr, σ)[0]
        traj.append(x_curr - v * dt_rev[i])

# 5. Plot shifted shapes and trajectories colored by absolute σ
fig, ax = plt.subplots(figsize=(8,8))

ax.scatter(x1[:,0], x1[:,1], s=3, alpha=0.2, color='black', label='GT samples')

# Color map over the *absolute* σ values in t_rev
norm = plt.Normalize(vmin=t_rev.min(), vmax=t_rev.max())
cmap = plt.cm.viridis

# Plot every trajectory point, colored by its true σ
for traj in trajectories:
    traj_arr = np.array(traj)
    ax.scatter(traj_arr[:,0], traj_arr[:,1],
               c=t_rev, cmap=cmap, norm=norm,
               s=5, linewidths=0)

# Mark initial noise points
ax.scatter(initial_points[:,0], initial_points[:,1],
           marker='*', s=80, color='red', label='Initial Points')

# Add a colorbar with the *absolute* σ range [1 → 0]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(t_rev)
cbar = fig.colorbar(sm, ax=ax, label='Sigma (absolute σ)')

ax.set_title('FM Sampling Trajectories\nColored by Absolute Sigma')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True)
ax.legend(fontsize='small')

plt.show()
