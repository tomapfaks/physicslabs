
import numpy as np
import matplotlib.pyplot as plt
 
 
def make_grid(start=-2, stop=2, numpoints=101):
    coords = np.linspace(start, stop, numpoints)
    x, y = np.meshgrid(coords, coords)
    return x, y
 
 
def scalar_field(x, y):
    """Z = sqrt(x^2 + y^2)."""
    return np.sqrt(x**2 + y**2)
 
 
def plot_contour_quiver(x, y, z, skip=5, fname='G2_contour_quiver.png'):
    dy, dx = np.gradient(z)  # np.gradient returns d/d(axis0), d/d(axis1)
 
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'2D function: $Z(x,y)=\sqrt{x^2+y^2}$ (contourf + quiver)')
 
    plt.contourf(x, y, z, 20)
    plt.set_cmap('coolwarm')
    plt.colorbar(label='Z')
 
    x_s, y_s = x[::skip, ::skip], y[::skip, ::skip]
    dx_s, dy_s = dx[::skip, ::skip], dy[::skip, ::skip]
    plt.quiver(x_s, y_s, dx_s, dy_s, color='k')  # auto-scaled arrow length
 
    plt.tight_layout()
    plt.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
 
 
def plot_contour_vs_contourf(x, y, z, fname='G2_contour_vs_contourf.png'):
    """Side-by-side comparison of contour() and contourf(), plus combined."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
 
    c0 = axes[0].contourf(x, y, z, 20, cmap='coolwarm')
    axes[0].set_title('contourf() only (filled)')
    fig.colorbar(c0, ax=axes[0], label='Z')
 
    c1 = axes[1].contour(x, y, z, 10, cmap='coolwarm')
    axes[1].clabel(c1, inline=True, fontsize=7)
    axes[1].set_title('contour() only (lines)')
 
    c2 = axes[2].contourf(x, y, z, 20, cmap='coolwarm')
    c3 = axes[2].contour(x, y, z, 10, colors='k', linewidths=0.6)
    axes[2].clabel(c3, inline=True, fontsize=7)
    axes[2].set_title('Both combined')
    fig.colorbar(c2, ax=axes[2], label='Z')
 
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
 
    fig.suptitle(r'$Z(x,y)=\sqrt{x^2+y^2}$: contourf vs. contour')
    fig.tight_layout()
    fig.savefig(f'/home/claude/graphical/{fname}', dpi=150, bbox_inches='tight')
    plt.close(fig)
 
 
def main():
    x, y = make_grid()
    z = scalar_field(x, y)
    plot_contour_quiver(x, y, z)
    plot_contour_vs_contourf(x, y, z)
    print('Saved G2_contour_quiver.png and G2_contour_vs_contourf.png')
 
 
if __name__ == '__main__':
    main()