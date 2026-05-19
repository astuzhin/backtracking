import backtracking_numba as bktg
import numpy as np
import pandas as pd

R_EARTH = 6771.2

# Start position in GTOD
r = R_EARTH * 1.2
lon = np.radians(0)
lat = np.radians(0)
colat = np.pi/2 - lat
x0_km = r * np.cos(lon) * np.sin(colat)
y0_km = r * np.sin(lon) * np.sin(colat)
z0_km = r * np.cos(colat)

# Start direction in GTOD
lon = np.radians(90)
colat = np.radians(70)
ux0 = np.cos(lon) * np.sin(colat)
uy0 = np.sin(lon) * np.sin(colat)
uz0 = np.cos(colat)

Energy = 2.0 # GeV

traj, n, status = bktg.trace_proton_backtracking(
        x0_km=x0_km, y0_km=y0_km, z0_km=z0_km,
        ux0=ux0, uy0=uy0, uz0=uz0,
        kinetic_energy_gev=Energy,
        charge=1.0,
        date=2025.0,
        stop_cond_mode=0,
        r_min_km=6371.2, # stop at 1 Earth radius
        r_max_km=10.0 * 6371.2, # stop at 10 Earth radii
        max_steps=20000,
        step_mode=1, # use dt0_s as the initial adaptive RK step
        ds0_km=1.0 # start with a step size of 1 km
)
traj = traj[:n+1]
traj = pd.DataFrame(traj, columns=['ts', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# Save example trajectory to csv (it is better to use uproot to save the trajectory)
traj.to_csv('traj.csv', index=False)


# Plot the trajectory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def draw_earth(ax):
    R = 6371.2
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='g', alpha=0.5)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
draw_earth(ax)
ax.plot(traj['x'], traj['y'], traj['z'], 'r-')
plt.show()