import numpy as np
import matplotlib.pyplot as plt

# 1. Define shifted mixture-of-manifold distribution for x1
def sample_x1(n_samples):
    centers = {
        0: np.array([5.0,  7.0]),  # circle center
        1: np.array([11.0, 7.0]),  # square center
        2: np.array([5.0, -7.0]),  # triangle center
        3: np.array([11.0, -7.0])  # gaussian center
    }
    samples = np.zeros((n_samples, 2))
    comps = np.random.choice(4, size=n_samples)
    for i, comp in enumerate(comps):
        c = centers[comp]
        if comp == 0:  # circle
            θ = np.random.uniform(0, 2*np.pi)
            samples[i] = c + np.array([np.cos(θ), np.sin(θ)])
        elif comp == 1:  # square
            edge = np.random.choice(4)
            t = np.random.uniform(-1, 1)
            if edge == 0:
                samples[i] = [c[0] + t, c[1] + 1]
            elif edge == 1:
                samples[i] = [c[0] + 1, c[1] + t]
            elif edge == 2:
                samples[i] = [c[0] + t, c[1] - 1]
            else:
                samples[i] = [c[0] - 1, c[1] + t]
        elif comp == 2:  # triangle
            R = 1.0
            h = np.sqrt(3) * R / 2
            v0 = c + np.array([0, 2*h/3])
            v1 = c + np.array([-R/2, -h/3])
            v2 = c + np.array([ R/2, -h/3])
            r1, r2 = np.random.rand(), np.random.rand()
            if r1 + r2 > 1:
                r1, r2 = 1-r1, 1-r2
            samples[i] = v0 + r1*(v1-v0) + r2*(v2-v0)
        else:  # gaussian
            samples[i] = np.random.multivariate_normal(c, 0.5**2*np.eye(2))
    return samples

# 2. Generate training data
N_data = 3000
x1     = sample_x1(N_data)
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
num_pts        = 50
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

# Draw the four manifolds
θ = np.linspace(0, 2*np.pi, 200)
ax.plot(5 + np.cos(θ), 7 + np.sin(θ), 'k-')           # Circle at (5,7)
sq = np.array([11,7])
ax.plot([sq[0]-1, sq[0]+1, sq[0]+1, sq[0]-1, sq[0]-1],
        [sq[1]-1, sq[1]-1, sq[1]+1, sq[1]+1, sq[1]-1], 'k-')  # Square
tri = np.array([5,-7]); h = np.sqrt(3)
verts = np.array([tri + np.array([0,2*h/3]),
                  tri + np.array([-1,-h/3]),
                  tri + np.array([ 1,-h/3])])
ax.plot(np.append(verts[:,0], verts[0,0]),
        np.append(verts[:,1], verts[0,1]), 'k-')       # Triangle
gauss_pts = np.random.multivariate_normal([11,-7], 0.5**2*np.eye(2), size=100)
ax.scatter(gauss_pts[:,0], gauss_pts[:,1], s=15, color='gray', alpha=0.5)

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
