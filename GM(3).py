# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:59:52 2026

@author: Toma
"""

import numpy as np
import matplotlib.pyplot as plt
 
 
def make_grid(start=-3, stop=3, numpoints=21):
    coords = np.linspace(start, stop, numpoints)
    x, v = np.meshgrid(coords, coords)
    return x, v
 
 
def sho_field(x, v, w=1.0):
    """dx/dt = v, dv/dt = -w^2 x."""
    dx = v
    dv = -(w**2) * x
    return dx, dv
 
 
def damped_field(x, v, b=0.3, w=1.0):
    """dx/dt = v, dv/dt = -b*v - w^2*x."""
    dx = v
    dv = -b * v - (w**2) * x
    return dx, dv
 
 
def plot_sho(fname='G3_sho.png'):
    x, v = make_grid()
    dx, dv = sho_field(x, v, w=1.0)
    energy = 0.5 * (v**2 + 1.0**2 * x**2)  # conserved for the SHO
 
    plt.figure(figsize=(6.5, 6.5))
    plt.gca().set_aspect('equal', adjustable='box')
 
    cs = plt.contour(x, v, energy, levels=8, cmap='viridis')
    plt.clabel(cs, inline=True, fontsize=7, fmt='E=%.1f')
 
    plt.quiver(x, v, dx, dv, pivot='mid', alpha=0.5)
    plt.streamplot(x, v, dx, dv, color='k', density=0.9, linewidth=0.8)
 
    plt.xlabel('x (position)')
    plt.ylabel('v (velocity)')
    plt.title(r'Simple harmonic oscillator phase space ($\omega$ = 1.0)')
    plt.tight_layout()
    plt.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
 
 
def plot_damped(fname='G3_damped.png'):
    x, v = make_grid()
    w = 1.0
    b_values = [0.2, 0.8, 2.5]  # under-, close to critically-, over-damped
    titles = ['Under-damped (b=0.2)', 'Near-critical (b=0.8)', 'Over-damped (b=2.5)']
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, b, title in zip(axes, b_values, titles):
        dx, dv = damped_field(x, v, b=b, w=w)
        ax.set_aspect('equal', adjustable='box')
        ax.quiver(x, v, dx, dv, pivot='mid', alpha=0.4)
        ax.streamplot(x, v, dx, dv, color=np.sqrt(dx**2 + dv**2),
                      cmap='plasma', density=0.9, linewidth=0.8)
        ax.plot(0, 0, 'ko', markersize=4)
        ax.set_xlabel('x (position)')
        ax.set_ylabel('v (velocity)')
        ax.set_title(f'{title}, $\\omega$={w}')
 
    fig.suptitle('Damped harmonic oscillator phase space for varying damping coefficient b')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def main():
    plot_sho()
    plot_damped()
    print('Saved G3_sho.png and G3_damped.png')
 
 
if __name__ == '__main__':
    main()
 