#!/usr/bin/env python3
"""
Magnetic Soft Continuum Robot (MSCR) Lung Navigation Simulation
===============================================================
Standalone simulation of an MSCR navigating through a bronchial tree phantom,
actuated by a UR5e-mounted permanent magnet. The inverse neural network maps
desired tip deflections to magnetic field parameters (|B|, azimuth, elevation),
which determine the UR5e end-effector (magnet) position.

Interactive: click on PRM nodes in the 3D view to set navigation targets.
Real-time graphs display NN output parameters and tip/magnet positions.
"""

import sys
import numpy as np
import scipy.io as sio
import onnxruntime as ort
from stl import mesh as stl_mesh
from collections import deque
import heapq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ──────────────────────────────────────────────────────────────────────────────
# FILE PATHS (adjust if needed)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR       = '/home/dozie/magnetcontrol_ws'
STL_PATH       = f'{BASE_DIR}/Bronchial tree anatomy-1mm-shell.STL'
PRM_PATH       = f'{BASE_DIR}/prm_roadmap.mat'
OBSTACLE_PATH  = f'{BASE_DIR}/bronchial_obstacle_map.mat'
ONNX_PATH      = f'{BASE_DIR}/src/magnet_control/magnet_control/mscr_inverse_model.onnx'
NORM_PATH      = f'{BASE_DIR}/src/magnet_control/magnet_control/inv_norm3.mat'

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
L_MSCR         = 60.0        # MSCR length [mm] (0.06 m)
MAGNET_OFFSET  = 80.0        # Nominal magnet distance from MSCR base [mm]
ALPHA_SMOOTH   = 0.15        # Exponential smoothing for magnet position
B_MIN, B_MAX   = 0.0005, 0.012  # |B| clamp range [T]
NAV_SPEED      = 2.0         # mm per simulation step along path
MM_TO_M        = 1e-3        # Conversion factor


class LungNavigationSim:
    """Main simulation class."""

    def __init__(self):
        print("[1/5] Loading neural network...")
        self._load_nn()

        print("[2/5] Loading PRM roadmap...")
        self._load_prm()

        print("[3/5] Loading bronchial obstacle map...")
        self._load_obstacles()

        print("[4/5] Loading lung STL mesh...")
        self._load_stl()

        print("[5/5] Initialising visualisation...")
        self._init_state()
        self._init_figures()

        print("\n" + "=" * 60)
        print("  MSCR LUNG NAVIGATION SIMULATION")
        print("=" * 60)
        print("  LEFT-CLICK  on the 3D lung view to set a target node.")
        print("  The robot will navigate via PRM shortest path.")
        print("  Close the window to exit.")
        print("=" * 60 + "\n")

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_nn(self):
        self.ort_sess = ort.InferenceSession(ONNX_PATH)
        mat = sio.loadmat(NORM_PATH)
        norm = mat['invNorm'][0, 0]
        self.mu_in   = norm['mu_in'].flatten().astype(np.float32)
        self.sig_in  = norm['sig_in'].flatten().astype(np.float32)
        self.mu_out  = norm['mu_out'].flatten().astype(np.float32)
        self.sig_out = norm['sig_out'].flatten().astype(np.float32)

    def _load_prm(self):
        prm = sio.loadmat(PRM_PATH)
        self.nodes = prm['nodesFiltered'].astype(np.float64)   # (N, 3) in mm
        self.edges = prm['edgesFiltered'].astype(np.int32) - 1  # 0-indexed

        # Build adjacency list with Euclidean weights
        N = len(self.nodes)
        self.adj = [[] for _ in range(N)]
        for e in self.edges:
            d = np.linalg.norm(self.nodes[e[0]] - self.nodes[e[1]])
            self.adj[e[0]].append((e[1], d))
            self.adj[e[1]].append((e[0], d))

        # Config
        cfg = prm['config'][0, 0]
        gi  = prm['gridInfo'][0, 0]
        self.grid_min = gi['minBounds'].flatten()
        self.grid_max = gi['maxBounds'].flatten()

    def _load_obstacles(self):
        obs = sio.loadmat(OBSTACLE_PATH)
        self.obstacle_map    = obs['obstacleMap']
        self.airway_interior = obs['airwayInterior']

    def _load_stl(self):
        lung_mesh = stl_mesh.Mesh.from_file(STL_PATH)
        self.stl_vectors = lung_mesh.vectors  # (n_tri, 3, 3) in mm
        # Subsample for rendering performance
        n_tri = len(self.stl_vectors)
        step = max(1, n_tri // 4000)
        self.stl_sub = self.stl_vectors[::step]

    # ── State ────────────────────────────────────────────────────────────────

    def _init_state(self):
        # Find a good starting node (highest Z ~ trachea entry)
        z_vals = self.nodes[:, 2]
        self.start_node = int(np.argmax(z_vals))
        self.current_pos = self.nodes[self.start_node].copy()
        self.target_node = None
        self.path_indices = []
        self.path_points  = np.empty((0, 3))
        self.path_cursor  = 0
        self.navigating   = False

        # MSCR base tracks a position behind the tip by L_MSCR
        self.mscr_base = self.current_pos.copy()
        self.mscr_base[2] += L_MSCR  # base is above tip (trachea direction)

        # Magnet (UR5e end-effector) position
        self.magnet_pos = self.current_pos + np.array([MAGNET_OFFSET, 0, 0])
        self.prev_magnet = self.magnet_pos.copy()

        # History buffers for graphs
        self.max_hist   = 300
        self.hist_B     = deque(maxlen=self.max_hist)
        self.hist_az    = deque(maxlen=self.max_hist)
        self.hist_el    = deque(maxlen=self.max_hist)
        self.hist_tipX  = deque(maxlen=self.max_hist)
        self.hist_tipY  = deque(maxlen=self.max_hist)
        self.hist_tipZ  = deque(maxlen=self.max_hist)
        self.hist_magX  = deque(maxlen=self.max_hist)
        self.hist_magY  = deque(maxlen=self.max_hist)
        self.hist_magZ  = deque(maxlen=self.max_hist)
        self.hist_t     = deque(maxlen=self.max_hist)
        self.step       = 0

    # ── Visualisation setup ──────────────────────────────────────────────────

    def _init_figures(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.suptitle('MSCR Lung Navigation — UR5e Magnetic Actuation',
                          fontsize=14, fontweight='bold')

        # Layout: left half = 3D scene, right column = 6 graphs
        gs = self.fig.add_gridspec(6, 2, width_ratios=[1.4, 1],
                                   hspace=0.45, wspace=0.30)

        # 3D lung view (spans all 6 rows on the left)
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection='3d')

        # Right column: 6 time-series graphs
        self.ax_B  = self.fig.add_subplot(gs[0, 1])
        self.ax_az = self.fig.add_subplot(gs[1, 1])
        self.ax_el = self.fig.add_subplot(gs[2, 1])
        self.ax_tx = self.fig.add_subplot(gs[3, 1])
        self.ax_ty = self.fig.add_subplot(gs[4, 1])
        self.ax_tz = self.fig.add_subplot(gs[5, 1])

        # Draw static STL mesh once
        self._draw_lung_mesh()

        # Connect mouse click
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _draw_lung_mesh(self):
        ax = self.ax3d
        poly = Poly3DCollection(self.stl_sub, alpha=0.06,
                                facecolor='lightskyblue', edgecolor='steelblue',
                                linewidth=0.1)
        ax.add_collection3d(poly)

        # Set axis limits from STL bounds
        mn = self.stl_vectors.reshape(-1, 3).min(axis=0)
        mx = self.stl_vectors.reshape(-1, 3).max(axis=0)
        pad = 10
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_zlim(mn[2] - pad, mx[2] + pad)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title('Bronchial Tree — click to set target')

    # ── PRM shortest path (Dijkstra) ────────────────────────────────────────

    def _dijkstra(self, src, dst):
        N = len(self.nodes)
        dist = np.full(N, np.inf)
        prev = np.full(N, -1, dtype=int)
        dist[src] = 0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[dst] == np.inf:
            return []
        path = []
        c = dst
        while c != -1:
            path.append(c)
            c = prev[c]
        return path[::-1]

    def _find_nearest_node(self, point_3d):
        """Find the PRM node closest to a 3D point."""
        dists = np.linalg.norm(self.nodes - point_3d, axis=1)
        return int(np.argmin(dists))

    # ── Neural network inference ─────────────────────────────────────────────

    def _nn_inference(self, tip_deflection_mm):
        """
        Run the inverse NN: tip deflection (mm) -> [|B|, azimuth, elevation].
        The NN was trained in meters, so we convert.
        """
        dptip = (tip_deflection_mm * MM_TO_M).astype(np.float32)
        x_norm = ((dptip - self.mu_in) / self.sig_in).reshape(1, -1)
        y_norm = self.ort_sess.run(None, {'input': x_norm})[0].flatten()
        y = y_norm * self.sig_out + self.mu_out
        return float(y[0]), float(y[1]), float(y[2])  # Bmag[T], az[rad], el[rad]

    def _compute_magnet_pos(self, Bmag, az, el, mscr_base_mm):
        """Map NN outputs to UR5e end-effector (magnet) position in mm."""
        # Radial distance: stronger B -> magnet closer
        R_mm = 40.0 + (1.0 - (np.clip(Bmag, B_MIN, B_MAX) / B_MAX)) * 60.0
        offset = np.array([
            R_mm * np.cos(el) * np.cos(az),
            R_mm * np.cos(el) * np.sin(az),
            R_mm * np.sin(el)
        ])
        pm_raw = mscr_base_mm + offset
        # Exponential smoothing
        pm = ALPHA_SMOOTH * pm_raw + (1.0 - ALPHA_SMOOTH) * self.prev_magnet
        self.prev_magnet = pm.copy()
        return pm

    # ── Mouse interaction ────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax3d:
            return
        # Project click to nearest PRM node by using the 2D screen coords
        # of all PRM nodes and finding the closest one to the click
        from mpl_toolkits.mplot3d import proj3d
        coords_2d = []
        for pt in self.nodes:
            x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2],
                                                self.ax3d.get_proj())
            coords_2d.append([x2, y2])
        coords_2d = np.array(coords_2d)

        # Convert click to display coordinates then to data coordinates
        click_display = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(coords_2d - click_display, axis=1)
        target_idx = int(np.argmin(dists))

        # Find current nearest node
        src_idx = self._find_nearest_node(self.current_pos)
        if target_idx == src_idx:
            return

        print(f"[NAV] Planning path: node {src_idx} -> node {target_idx}")
        path = self._dijkstra(src_idx, target_idx)
        if not path:
            print("[NAV] No path found!")
            return

        # Interpolate path to finer resolution for smooth navigation
        self.path_indices = path
        raw_pts = self.nodes[path]
        self.path_points = self._interpolate_path(raw_pts, step_mm=NAV_SPEED)
        self.path_cursor = 0
        self.navigating = True
        self.target_node = target_idx
        print(f"[NAV] Path has {len(path)} PRM nodes, "
              f"{len(self.path_points)} interpolated waypoints.")

    def _interpolate_path(self, points, step_mm=2.0):
        """Resample polyline to uniform step size."""
        if len(points) < 2:
            return points
        interp = [points[0]]
        residual = 0.0
        for i in range(1, len(points)):
            seg = points[i] - points[i - 1]
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-9:
                continue
            direction = seg / seg_len
            travelled = residual
            while travelled < seg_len:
                interp.append(points[i - 1] + direction * travelled)
                travelled += step_mm
            residual = travelled - seg_len
        interp.append(points[-1])
        return np.array(interp)

    # ── Simulation loop ──────────────────────────────────────────────────────

    def run(self):
        try:
            while plt.fignum_exists(self.fig.number):
                if self.navigating:
                    self._navigation_step()
                self._update_plots()
                plt.pause(0.03)
        except KeyboardInterrupt:
            pass
        print("\nSimulation ended.")
        plt.close('all')

    def _navigation_step(self):
        if self.path_cursor >= len(self.path_points):
            self.navigating = False
            print("[NAV] Target reached!")
            return

        # Advance along path
        self.current_pos = self.path_points[self.path_cursor].copy()
        self.path_cursor += 1

        # MSCR base is L_MSCR behind tip along the path tangent
        if self.path_cursor >= 2:
            tangent = self.current_pos - self.path_points[max(0, self.path_cursor - 2)]
            t_norm = np.linalg.norm(tangent)
            if t_norm > 1e-6:
                tangent /= t_norm
            else:
                tangent = np.array([0, 0, 1.0])
        else:
            tangent = np.array([0, 0, 1.0])
        self.mscr_base = self.current_pos - tangent * L_MSCR

        # Compute tip deflection relative to straight MSCR
        straight_tip = self.mscr_base + tangent * L_MSCR
        deflection_mm = self.current_pos - straight_tip

        # If deflection is essentially zero, inject small perturbation for NN
        if np.linalg.norm(deflection_mm) < 0.01:
            deflection_mm = np.array([0.1, 0.1, 0.05])

        # NN inference
        Bmag, az, el = self._nn_inference(deflection_mm)

        # Compute magnet position
        self.magnet_pos = self._compute_magnet_pos(Bmag, az, el, self.mscr_base)

        # Record history
        self.step += 1
        self.hist_t.append(self.step)
        self.hist_B.append(Bmag * 1000)  # mT
        self.hist_az.append(np.degrees(az))
        self.hist_el.append(np.degrees(el))
        self.hist_tipX.append(self.current_pos[0])
        self.hist_tipY.append(self.current_pos[1])
        self.hist_tipZ.append(self.current_pos[2])
        self.hist_magX.append(self.magnet_pos[0])
        self.hist_magY.append(self.magnet_pos[1])
        self.hist_magZ.append(self.magnet_pos[2])

    # ── Plotting ─────────────────────────────────────────────────────────────

    def _update_plots(self):
        # ── 3D scene ──
        ax = self.ax3d
        # Remove previous dynamic artists (keep mesh)
        while len(ax.lines) > 0:
            ax.lines[0].remove()
        while len(ax.collections) > 1:  # keep the STL Poly3D (index 0)
            ax.collections[-1].remove()

        # Draw PRM nodes (small, grey)
        ax.scatter(self.nodes[::3, 0], self.nodes[::3, 1], self.nodes[::3, 2],
                   c='grey', s=1, alpha=0.3)

        # Draw planned path
        if len(self.path_points) > 0:
            ax.plot(self.path_points[:, 0], self.path_points[:, 1],
                    self.path_points[:, 2], 'lime', linewidth=2, label='PRM path')

        # Draw traversed path
        if self.path_cursor > 1:
            trav = self.path_points[:self.path_cursor]
            ax.plot(trav[:, 0], trav[:, 1], trav[:, 2],
                    'r-', linewidth=2.5, label='Traversed')

        # MSCR body (base -> tip)
        ax.plot([self.mscr_base[0], self.current_pos[0]],
                [self.mscr_base[1], self.current_pos[1]],
                [self.mscr_base[2], self.current_pos[2]],
                'r-', linewidth=4)

        # Tip marker
        ax.scatter(*self.current_pos, color='red', s=100, zorder=5,
                   depthshade=False, label='MSCR tip')

        # Magnet (UR5e end-effector)
        ax.scatter(*self.magnet_pos, color='black', s=120, marker='D',
                   zorder=5, depthshade=False, label='UR5e magnet')

        # Dashed line: magnet -> tip (magnetic field direction)
        ax.plot([self.magnet_pos[0], self.current_pos[0]],
                [self.magnet_pos[1], self.current_pos[1]],
                [self.magnet_pos[2], self.current_pos[2]],
                'k--', linewidth=0.8, alpha=0.5)

        # Start node
        ax.scatter(*self.nodes[self.start_node], color='blue', s=80,
                   marker='^', depthshade=False, label='Start')

        # Target node
        if self.target_node is not None:
            ax.scatter(*self.nodes[self.target_node], color='gold', s=120,
                       marker='*', depthshade=False, label='Target')

        ax.set_title('Bronchial Tree Navigation — click to set target')
        ax.legend(loc='upper left', fontsize=7, markerscale=0.6)

        # ── Time-series graphs ──
        t = list(self.hist_t)
        if len(t) == 0:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        def _plot_ts(ax_ts, data, ylabel, color, label):
            ax_ts.cla()
            ax_ts.plot(t, list(data), color=color, linewidth=1.2)
            ax_ts.set_ylabel(ylabel, fontsize=8)
            ax_ts.tick_params(labelsize=7)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.set_xlim(max(0, t[-1] - self.max_hist), t[-1] + 5)

        _plot_ts(self.ax_B,  self.hist_B,  '|B| [mT]',        '#e74c3c', '|B|')
        _plot_ts(self.ax_az, self.hist_az, 'Azimuth [deg]',    '#3498db', 'Az')
        _plot_ts(self.ax_el, self.hist_el, 'Elevation [deg]',  '#2ecc71', 'El')

        # Tip position
        self.ax_tx.cla()
        self.ax_tx.plot(t, list(self.hist_tipX), '#e74c3c', linewidth=1, label='Tip X')
        self.ax_tx.plot(t, list(self.hist_magX), '#e74c3c', linewidth=1, linestyle='--', label='Mag X')
        self.ax_tx.set_ylabel('X [mm]', fontsize=8)
        self.ax_tx.legend(fontsize=6, loc='upper left')
        self.ax_tx.tick_params(labelsize=7)
        self.ax_tx.grid(True, alpha=0.3)
        self.ax_tx.set_xlim(max(0, t[-1] - self.max_hist), t[-1] + 5)

        self.ax_ty.cla()
        self.ax_ty.plot(t, list(self.hist_tipY), '#3498db', linewidth=1, label='Tip Y')
        self.ax_ty.plot(t, list(self.hist_magY), '#3498db', linewidth=1, linestyle='--', label='Mag Y')
        self.ax_ty.set_ylabel('Y [mm]', fontsize=8)
        self.ax_ty.legend(fontsize=6, loc='upper left')
        self.ax_ty.tick_params(labelsize=7)
        self.ax_ty.grid(True, alpha=0.3)
        self.ax_ty.set_xlim(max(0, t[-1] - self.max_hist), t[-1] + 5)

        self.ax_tz.cla()
        self.ax_tz.plot(t, list(self.hist_tipZ), '#2ecc71', linewidth=1, label='Tip Z')
        self.ax_tz.plot(t, list(self.hist_magZ), '#2ecc71', linewidth=1, linestyle='--', label='Mag Z')
        self.ax_tz.set_ylabel('Z [mm]', fontsize=8)
        self.ax_tz.set_xlabel('Step', fontsize=8)
        self.ax_tz.legend(fontsize=6, loc='upper left')
        self.ax_tz.tick_params(labelsize=7)
        self.ax_tz.grid(True, alpha=0.3)
        self.ax_tz.set_xlim(max(0, t[-1] - self.max_hist), t[-1] + 5)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sim = LungNavigationSim()
    sim.run()
