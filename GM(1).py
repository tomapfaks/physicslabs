import numpy as np
import matplotlib.pyplot as plt
 
 
def make_grid(start=0, stop=2 * np.pi, numpoints=101):
    """Creates a square grid of (x, y) points using linspace + meshgrid.
 
    linspace is used (rather than arange) because the brief specifies an
    exact number of grid points (101) spanning the interval inclusive of
    both endpoints; arange would require working out a step size by hand
    and risks including/excluding the endpoint due to floating-point
    rounding, as noted in G0_ranges.py.
    """
    points = np.linspace(start, stop, numpoints)
    x, y = np.meshgrid(points, points)
    return x, y
 
 
def vector_field(x, y):
    """Vx = cos(x)*y, Vy = sin(x)*x."""
    vx = np.cos(x) * y
    vy = np.sin(x) * x
    return vx, vy
 
 
def plot_quiver(x, y, vx, vy, reduction=5, fname='G1_quiver.png'):
    plt.figure(figsize=(6.5, 6.5))
    plt.gca().set_aspect('equal', adjustable='box')
 
    # show a vector on every 5th grid point in each direction
    xr = x[::reduction, ::reduction]
    yr = y[::reduction, ::reduction]
    vxr = vx[::reduction, ::reduction]
    vyr = vy[::reduction, ::reduction]
 
    q = plt.quiver(xr, yr, vxr, vyr, pivot='mid',
                    label=r'$V_x=\cos(x)y,\ V_y=\sin(x)x$')
    plt.quiverkey(q, 0.82, -0.09, 5, r'$5$', labelpos='E', coordinates='axes')
 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Quiver plot of $V(x,y) = (\cos(x)y,\ \sin(x)x)$')
    plt.legend(loc='upper left', fontsize=8)
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
 
 
def main():
    x, y = make_grid()
    vx, vy = vector_field(x, y)
    plot_quiver(x, y, vx, vy)
    print('Saved G1_quiver.png')
 
 
if __name__ == '__main__':
    main()