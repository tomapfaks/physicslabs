import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
 
 
def lv_field(x, y, a, b, c, d):
    dx = a * x - b * x * y
    dy = c * x * y - d * y
    return dx, dy
 
 
def solve_lv(a, b, c, d, x0, y0, tmax=30, n_points=1000):
    t_eval = np.linspace(0, tmax, n_points)
 
    def rhs(t, state):
        x, y = state
        return list(lv_field(x, y, a, b, c, d))
 
    sol = solve_ivp(rhs, (0, tmax), [x0, y0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]
 
 
def plot_quiver_streamline(a, b, c, d, x0, y0, fname='G5_quiver_streamline.png'):
    """Quiver/streamline plot of the vector field, with a single
    streamline starting at (x0, y0)."""
    coords_x = np.linspace(0.1, 8, 25)
    coords_y = np.linspace(0.1, 6, 25)
    x, y = np.meshgrid(coords_x, coords_y)
    dx, dy = lv_field(x, y, a, b, c, d)
 
    plt.figure(figsize=(7, 6))
    plt.quiver(x, y, dx, dy, alpha=0.4)
    seed = np.array([[x0], [y0]])
    plt.streamplot(x, y, dx, dy, color='b', start_points=seed.T, linewidth=1.5)
    plt.plot(x0, y0, 'bo', label=f'$x_0$={x0}, $y_0$={y0}')
    plt.xlabel('rabbits')
    plt.ylabel('foxes')
    plt.title(f'Lotka-Volterra vector field (a={a}, b={b}, c={c:.3g}, d={d})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
 
 
def plot_time_and_phase(a, b, c, d, x0, y0, fname='G5_time_and_phase.png', tmax=30):
    t, x, y = solve_lv(a, b, c, d, x0, y0, tmax=tmax)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t, x, 'b-', label='rabbits')
    ax1.plot(t, y, 'g-.', label='foxes')
    ax1.set_xlabel('time')
    ax1.set_ylabel('number')
    ax1.set_title(f'Time series ($x_0$={x0}, $y_0$={y0})')
    ax1.legend(loc='upper right')
 
    ax2.plot(x, y, 'r')
    ax2.set_xlabel('rabbits')
    ax2.set_ylabel('foxes')
    ax2.set_title('Phase space')
 
    fig.suptitle(f'Lotka-Volterra model: a={a}, b={b}, c={c:.3g}, d={d}')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def plot_multiple_ic(a, b, c, d, ic_list, fname='G5_multiple_ic.png', tmax=30):
    """x(t)/y(t) and phase plots for several initial conditions, same params."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for x0, y0 in ic_list:
        t, x, y = solve_lv(a, b, c, d, x0, y0, tmax=tmax)
        ax1.plot(t, x, label=f'rabbits, $x_0$={x0},$y_0$={y0}')
        ax1.plot(t, y, '--', label=f'foxes, $x_0$={x0},$y_0$={y0}')
        ax2.plot(x, y, label=f'$x_0$={x0}, $y_0$={y0}')
    ax1.set_xlabel('time'); ax1.set_ylabel('number')
    ax1.set_title('Time series for different initial conditions')
    ax1.legend(fontsize=7, ncol=2)
    ax2.set_xlabel('rabbits'); ax2.set_ylabel('foxes')
    ax2.set_title('Phase space for different initial conditions')
    ax2.legend(fontsize=8)
    fig.suptitle(f'Lotka-Volterra model: a={a}, b={b}, c={c:.3g}, d={d} (fixed parameters)')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def plot_multiple_params(param_list, x0, y0, fname='G5_multiple_params.png', tmax=30):
    """x(t)/y(t) and phase plots for several parameter sets, same ICs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for a, b, c, d in param_list:
        t, x, y = solve_lv(a, b, c, d, x0, y0, tmax=tmax)
        label = f'a={a},b={b},c={c:.2g},d={d}'
        ax1.plot(t, x, label=f'rabbits, {label}')
        ax2.plot(x, y, label=label)
    ax1.set_xlabel('time'); ax1.set_ylabel('rabbits')
    ax1.set_title('Rabbit population for different parameters')
    ax1.legend(fontsize=7)
    ax2.set_xlabel('rabbits'); ax2.set_ylabel('foxes')
    ax2.set_title('Phase space for different parameters')
    ax2.legend(fontsize=7)
    fig.suptitle(f'Lotka-Volterra model: $x_0$={x0}, $y_0$={y0} (fixed initial conditions)')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def main():
    a, b, c, d = 4.0, 2.0, 1 / 3, 1.0
    x0, y0 = 4.0, 2.0
 
    plot_quiver_streamline(a, b, c, d, x0, y0)
    plot_time_and_phase(a, b, c, d, x0, y0)
 
    plot_multiple_ic(a, b, c, d,
                      ic_list=[(4.0, 2.0), (2.0, 1.0), (6.0, 3.0), (1.0, 4.0)],
                      fname='G5_multiple_ic.png')
 
    plot_multiple_params([
        (4.0, 2.0, 1/3, 1.0),
        (4.0, 1.0, 1/3, 1.0),   # weaker predation
        (4.0, 2.0, 1/3, 2.0),   # faster fox death
        (6.0, 2.0, 1/3, 1.0),   # faster rabbit growth
    ], x0, y0, fname='G5_multiple_params.png')
 
    print('Saved G5_quiver_streamline.png, G5_time_and_phase.png, '
          'G5_multiple_ic.png, G5_multiple_params.png')
 
 
if __name__ == '__main__':
    main()