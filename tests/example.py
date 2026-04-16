import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_helper import plot_plane_3d, plot_voxel_walls_3d, plot_fibres
from utils import *

import time

# initialise grid
grid = grid_spec(1.0, 2, 3, 0.03, 0.03, 0.03)

# generate fibres
fibres = fibre([[0.2, 0.5, 0.5]], # ref point
               [[1.0, 1.0, 1.0]], # fibre vector
               [0.3])                         # fibre radius

## visualise voxel centers and fibre center lines
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=centers[:,1], s=1)
#ax.set_box_aspect([grid.Lx/grid.Lz,grid.Ly/grid.Lz,1])
#plot_fibres(fibres.x0, fibres.v, ax=ax, color='red', point_color='black', point_size=50, linewidth=5)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()

#
centers = np.array(grid.centers())
x0 = np.array(fibres.x0)
v = np.array(fibres.v)
r = np.array(fibres.r)
h_vec = np.array([grid.dx, grid.dy, grid.dz])

###########################################
## associate voxels to the closest fibre ##
###########################################

all_in, all_out, composite, idx_closest = fibre_association(centers, h_vec, x0, v, r)
matID = np.where(all_out, 0, idx_closest + 1)

# visualise voxel centers colored by fibre association
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=matID, s=20, cmap='tab10')
plot_fibres(fibres.x0, fibres.v, ax=ax, color='red', point_color='black', point_size=50, linewidth=5)
ax.set_box_aspect([grid.Lx/grid.Lz,grid.Ly/grid.Lz,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.show()

###########################
## Clip composite voxels ##
###########################

centers_cv = centers[composite]

# intersection plane (defined by a point and a normal vector)
cutplane_P0, cutplane_n = reconstruct_plane(centers_cv, x0, v, r)

# level set values at vertices of voxels
phi = get_level_set_at_points(centers_cv, h_vec, cutplane_P0, cutplane_n, VERT=None)

# patching a potential issue (due to inconsistency between curved fibre surface and intersection plane)
mask_issue = np.all(phi<0, axis=1)
idx_issue = np.where(mask_issue)[0]
print(f'rejecting {len(idx_issue)} composite voxels')
for i in idx_issue:
    idx = np.where( (centers == centers_cv[i]).all(axis=1))[0][0]
    matID[idx] = idx_closest[idx] + 1  #TODO: this may not be necessary!!!
centers_cv = centers_cv[~mask_issue]
phi = phi[~mask_issue]
cutplane_P0 = cutplane_P0[~mask_issue]
cutplane_n = cutplane_n[~mask_issue]

# compute polyhedron volume and intersection area 
vols, areas = voxel_clipping(centers_cv, h_vec, phi)


################################################################
## verification: plane reconstruction and polyhedron vertices ##
################################################################

ivx = 12 # index of composite voxel of your choice

cutplane_P0 = cutplane_P0[ivx]
cutplane_n = cutplane_n[ivx]
cutPts, infibres = get_polyhedron_vertices(centers_cv[ivx], h_vec, phi[ivx])
cutPts = order_polygon_vertices(cutPts)
cutPts = np.concatenate([cutPts, cutPts[0:1]], axis=0)

ax = plot_voxel_walls_3d(
        centers_cv[[ivx]],
        voxel_size=(grid.dx, grid.dy, grid.dz),
        face_alpha=0.20,
        dot_size=10,
        title="Semi-transparent voxel walls + center dots"
    )
plot_fibres(fibres.x0, fibres.v, ax=ax, color='red', point_color='black', point_size=50, linewidth=5)
ax.plot(cutplane_P0[0], cutplane_P0[1], cutplane_P0[2], linestyle='None', marker='o', color='blue', markersize=5, label='Interface Points')
ax.quiver(
    cutplane_P0[0], cutplane_P0[1], cutplane_P0[2],
    cutplane_n[0], cutplane_n[1], cutplane_n[2],
    length=0.2, color='orange', label='Interface Normals'
)
plot_plane_3d(cutplane_P0, cutplane_n, size=max(h_vec), ax=ax, 
              plane_color='cyan', plane_alpha=0.5, point_color='blue', 
              vector_color='orange', vector_length=0.5)
ax.scatter(cutPts[:,0], cutPts[:,1], cutPts[:,2], color='magenta', s=50, label='Polyhedron Vertices')
ax.scatter(infibres[:,0], infibres[:,1], infibres[:,2], color='green', s=50, label='In-fibre Vertices')
ax.plot(*cutPts.T, '--')
ax.set_axis_off
print(f'intersection plane surface area: {areas[ivx]}')
print(f'polyhedron volume: {vols[ivx]}')
plt.show()

