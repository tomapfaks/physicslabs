# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:00:19 2026

@author: Toma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
 
 
def verhulst_rate(x, r, K):
    """x' = r*x*(1 - x/K)."""
    return r * x * (1 - x / K)
 
 
def plot_fixed_points(r, K, x_range=None, fname='G4_fixed_points.png'):
    """Plot x' vs x and mark/classify the fixed points with flow arrows."""
    if x_range is None:
        x_range = (-0.3 * K, 1.5 * K)
    x = np.linspace(*x_range, 400)
    xdot = verhulst_rate(x, r, K)
 
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.plot(x, xdot, 'b-', label=f"$x'=rx(1-x/K)$, r={r}, K={K}")
 
    fixed_points = [0, K]
    for fp in fixed_points:
        # determine stability: sign of d(xdot)/dx just either side of fp
        eps = 1e-3 * K if K != 0 else 1e-3
        slope_left = verhulst_rate(fp - eps, r, K)
        slope_right = verhulst_rate(fp + eps, r, K)
        stable = slope_left > 0 and slope_right < 0
        colour = 'green' if stable else 'red'
        marker_label = 'stable' if stable else 'unstable'
        ax.plot(fp, 0, 'o', color=colour, markersize=10, zorder=5)
        ax.annotate(f'x={fp:g}\n({marker_label})', (fp, 0),
                    textcoords='offset points', xytext=(0, 18 if r > 0 else -28),
                    ha='center', fontsize=9, color=colour)
 
    # flow-on-a-line arrows just above the x-axis, coloured by direction
    arrow_x = np.linspace(x_range[0] + 0.05 * (x_range[1] - x_range[0]),
                           x_range[1] - 0.05 * (x_range[1] - x_range[0]), 14)
    arrow_rate = verhulst_rate(arrow_x, r, K)
    arrow_dir = np.sign(arrow_rate)
    y0 = (xdot.max() - xdot.min()) * -0.12 + xdot.min() * 0  # just below axis line visually
    for ax_x, d in zip(arrow_x, arrow_dir):
        if d == 0:
            continue
        ax.annotate('', xy=(ax_x + 0.03 * d * (x_range[1] - x_range[0]), 0),
                     xytext=(ax_x, 0),
                     arrowprops=dict(arrowstyle='-|>', color='k'))
 
    ax.set_xlabel('x (population)')
    ax.set_ylabel("x' (growth rate)")
    ax.set_title(f'Flow-on-a-line: fixed points of the Verhulst model (r={r}, K={K})')
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def solve_verhulst(r, K, x0, tmax=20, n_points=300):
    t_eval = np.linspace(0, tmax, n_points)
    sol = solve_ivp(lambda t, x: verhulst_rate(x, r, K), (0, tmax), [x0], t_eval=t_eval)
    return sol.t, sol.y[0]
 
 
def plot_time_series(r, K, x0_list, fname='G4_time_series.png', tmax=20):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for x0 in x0_list:
        t, x = solve_verhulst(r, K, x0, tmax=tmax)
        ax.plot(t, x, label=f'$x_0$={x0:g}')
    ax.axhline(K, color='grey', linestyle='--', linewidth=1, label=f'K={K}')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xlabel('t')
    ax.set_ylabel('x(t) (population)')
    ax.set_title(f'Verhulst model solutions for various $x_0$ (r={r}, K={K})')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def main():
    # (a) fixed points for the standard growth case r > 0, K > 0
    plot_fixed_points(r=0.5, K=10, fname='G4_fixed_points.png')
 
    # (a, continued) a decay case r < 0, to show the roles reverse
    plot_fixed_points(r=-0.3, K=10, x_range=(-5, 20), fname='G4_fixed_points_negative_r.png')
 
    # (b) time series for several physically meaningful initial conditions:
    # between 0 and K, exactly at K, and above K
    plot_time_series(r=0.5, K=10, x0_list=[0.5, 2, 5, 10, 15],
                      fname='G4_time_series.png')
 
    # a separate plot showing the unphysical x0 < 0 case, which diverges
    # to -infinity in finite time (consistent with x=0 being unstable from
    # the left) -- shown on its own so it doesn't swamp the y-axis above
    plot_time_series(r=0.5, K=10, x0_list=[-0.5, -1, -2],
                      fname='G4_time_series_negative_x0.png', tmax=4)
 
    # explore a different r, K combination
    plot_time_series(r=1.2, K=50, x0_list=[1, 10, 50, 80],
                      fname='G4_time_series_r1.2_K50.png')
 
    print('Saved G4_fixed_points.png, G4_fixed_points_negative_r.png, '
          'G4_time_series.png, G4_time_series_r1.2_K50.png')
 
 
if __name__ == '__main__':
    main()