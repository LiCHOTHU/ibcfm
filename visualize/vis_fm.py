import numpy as np
import matplotlib.pyplot as plt

# Projection functions for manifold convergence
def project_circle(x, center, R=1.0):
    rel = x - center
    norm = np.linalg.norm(rel)
    return center + (rel / norm) * R if norm != 0 else center + np.array([R, 0])

def project_square(x, center, R=1.0):
    rel = x - center
    dx, dy = rel
    if abs(dx) > abs(dy):
        sign = np.sign(dx) if dx != 0 else 1
        return center + np.array([sign * R, np.clip(dy, -R, R)])
    else:
        sign = np.sign(dy) if dy != 0 else 1
        return center + np.array([np.clip(dx, -R, R), sign * R])

def project_triangle(x, center, R=1.0):
    h = np.sqrt(3) * R / 2
    verts = [
        center + np.array([0, 2*h/3]),
        center + np.array([-R/2, -h/3]),
        center + np.array([R/2, -h/3])
    ]
    best_pt, best_dist = None, np.inf
    for i in range(3):
        v0, v1 = verts[i], verts[(i+1)%3]
        seg = v1 - v0
        t = np.dot(x - v0, seg) / np.dot(seg, seg)
        proj = v0 + np.clip(t, 0, 1) * seg
        dist = np.linalg.norm(x - proj)
        if dist < best_dist:
            best_dist, best_pt = dist, proj
    return best_pt

# Sample for Gaussian manifold
gauss_center = np.array([3.0, -3.0])
np.random.seed(0)
gauss_samples = np.random.multivariate_normal(gauss_center, 0.5**2 * np.eye(2), size=40)

# Vector field with correct three-stage behavior
def vector_field(x, sigma, mu,
                 circle_c, square_c, triangle_c, gauss_c,
                 sigma_high, sigma_low):
    if sigma > sigma_high:
        # Stage 1: global mean attraction
        return -(x - mu) / sigma
    elif sigma > sigma_low:
        # Stage 2: cluster attraction to the nearest cluster center
        centers = [circle_c[0], square_c[0], triangle_c[0], gauss_c]
        dists = [np.linalg.norm(x - c) for c in centers]
        nearest = centers[int(np.argmin(dists))]
        return (nearest - x) / sigma
    else:
        # Stage 3: manifold convergence (snap to nearest boundary or sample)
        candidates = [
            project_circle(x, circle_c[0]),
            project_square(x, square_c[0]),
            project_triangle(x, triangle_c[0])
        ] + list(gauss_samples)
        y_star = min(candidates, key=lambda y: np.linalg.norm(y - x))
        return (y_star - x) / sigma

# Manifold and initial setup
circle_centers = [np.array([-3.0,  3.0])]
square_centers = [np.array([ 3.0,  3.0])]
triangle_centers = [np.array([-3.0, -3.0])]
mu = (circle_centers[0] + square_centers[0] + triangle_centers[0] + gauss_center) / 4

initial_points = [
    np.array([-7.0,  7.0]), np.array([ 7.0,  7.0]),
    np.array([-7.0, -7.0]), np.array([ 7.0, -7.0]),
    np.array([ 0.0,  8.0]), np.array([ 0.0, -8.0])
]

# Time (sigma) schedule and steps
t_vals = np.linspace(8.0, 0.01, 300)
dt_vals = -np.diff(t_vals)  # positive ∆σ
sigma_high, sigma_low = 4.0, 1.0

# Integrate trajectories
trajectories = []
stages_list = []
for x0 in initial_points:
    x = x0.copy()
    traj = [x.copy()]
    stages = []
    for i, sigma in enumerate(t_vals[:-1]):
        stage = 1 if sigma > sigma_high else 2 if sigma > sigma_low else 3
        stages.append(stage)
        v = vector_field(x, sigma, mu,
                         circle_centers, square_centers,
                         triangle_centers, gauss_center,
                         sigma_high, sigma_low)
        x = x + dt_vals[i] * v
        traj.append(x.copy())
    trajectories.append(np.array(traj))
    stages_list.append(np.array(stages))

# Plotting
plt.figure(figsize=(8,8))
theta = np.linspace(0, 2*np.pi, 200)
# Draw manifolds
plt.plot(circle_centers[0][0] + np.cos(theta),
         circle_centers[0][1] + np.sin(theta), 'k-', label='Circle')
sq = square_centers[0]
plt.plot([sq[0]-1, sq[0]+1, sq[0]+1, sq[0]-1, sq[0]-1],
         [sq[1]-1, sq[1]-1, sq[1]+1, sq[1]+1, sq[1]-1], 'k-', label='Square')
tri = triangle_centers[0]; h=np.sqrt(3)
verts = [tri + np.array([0,2*h/3]), tri + np.array([-1,-h/3]), tri + np.array([1,-h/3])]
plt.plot([v[0] for v in verts+[verts[0]]],
         [v[1] for v in verts+[verts[0]]], 'k-', label='Triangle')
plt.scatter(gauss_samples[:,0], gauss_samples[:,1], s=20, color='gray', alpha=0.6, label='Gaussian Cloud')

# Initial points
for idx, p in enumerate(initial_points):
    plt.scatter(p[0], p[1], marker='*', s=100,
                color='black', label='Initial Points' if idx==0 else "")

# Plot trajectories and stage markers
markers = {1:'o', 2:'s', 3:'^'}
colors  = {1:'blue', 2:'green', 3:'red'}
labels  = {1:'Stage 1: Mean', 2:'Stage 2: Cluster', 3:'Stage 3: Manifold'}
for traj, stages in zip(trajectories, stages_list):
    # full path
    plt.plot(traj[:,0], traj[:,1], color='lightgray', linewidth=1)
    # stage points
    for stage in [1,2,3]:
        mask = stages == stage
        pts = traj[:-1][mask]
        plt.scatter(pts[:,0], pts[:,1], color=colors[stage],
                    marker=markers[stage], s=20,
                    label=labels[stage] if traj is trajectories[0] else "")

plt.title('Corrected FM Visualization: Proper Three-Stage Drift')
plt.xlabel('x'); plt.ylabel('y')
plt.axis('equal')
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.show()
