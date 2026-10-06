import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
 
 
def field(x, y):
    dx = y
    dy = -x + (1 - x**2) * y
    return dx, dy
 
 
def solve(x0, y0, tmax=30, n_points=2000):
    t_eval = np.linspace(0, tmax, n_points)
 
    def rhs(t, state):
        x, y = state
        return list(field(x, y))
 
    sol = solve_ivp(rhs, (0, tmax), [x0, y0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]
 
 
def plot_quiver_stream(fname='G6_quiver_stream.png'):
    coords = np.linspace(-4, 4, 25)
    x, y = np.meshgrid(coords, coords)
    dx, dy = field(x, y)
 
    plt.figure(figsize=(6.5, 6.5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.quiver(x, y, dx, dy, alpha=0.4)
    plt.streamplot(x, y, dx, dy, color='k', density=1.1, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r"Vector field: $\dot{x}=y,\ \dot{y}=-x+(1-x^2)y$")
    plt.tight_layout()
    plt.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
 
 
def plot_time_series(ic_list, fname='G6_time_series.png', tmax=30):
    fig, axes = plt.subplots(len(ic_list), 1, figsize=(8, 2.3 * len(ic_list)), sharex=True)
    for ax, (x0, y0) in zip(axes, ic_list):
        t, x, y = solve(x0, y0, tmax=tmax)
        ax.plot(t, x, label='x(t)')
        ax.plot(t, y, label='y(t)')
        ax.set_ylabel('x, y')
        ax.set_title(f'$x_0$={x0}, $y_0$={y0}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('t')
    fig.suptitle('Time series for different initial conditions')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def plot_phase_with_isoclines(ic_list, fname='G6_phase_isoclines.png', tmax=30):
    fig, ax = plt.subplots(figsize=(7.5, 7))
 
    for x0, y0 in ic_list:
        t, x, y = solve(x0, y0, tmax=tmax)
        ax.plot(x, y, label=f'$x_0$={x0}, $y_0$={y0}')
        ax.plot(x0, y0, 'o', markersize=4)
 
    # x-isocline: dx/dt = y = 0  ->  the line y = 0
    x_range = np.linspace(-4, 4, 400)
    ax.axhline(0, color='k', linestyle='--', linewidth=1.3, label='x-isocline: y=0')
 
    # y-isocline: dy/dt = -x + (1-x^2)y = 0  ->  y = x / (1 - x^2), x != +-1
    x_iso = np.linspace(-4, 4, 2000)
    with np.errstate(divide='ignore', invalid='ignore'):
        y_iso = x_iso / (1 - x_iso**2)
    # mask the asymptotes near x = +-1 so the plot doesn't draw vertical spikes
    y_iso_masked = np.where(np.abs(1 - x_iso**2) > 0.05, y_iso, np.nan)
    ax.plot(x_iso, y_iso_masked, 'm--', linewidth=1.3, label='y-isocline: y=x/(1-x\u00b2)')
 
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase-space trajectories and isoclines (limit cycle)')
    ax.legend(fontsize=8, loc='upper right', ncol=1)
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def main():
    ic_list = [(3, 3), (-2, 2), (0.1, 0.1), (0.5, 1), (4, -3), (-0.1, -0.1)]
 
    plot_quiver_stream()
    plot_time_series(ic_list[:4])
    plot_phase_with_isoclines(ic_list)
 
    print('Saved G6_quiver_stream.png, G6_time_series.png, G6_phase_isoclines.png')
 
 
if __name__ == '__main__':
    main()