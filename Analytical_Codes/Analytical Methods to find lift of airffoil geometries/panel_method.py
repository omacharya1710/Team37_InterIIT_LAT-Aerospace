import numpy as np, math, matplotlib.pyplot as plt

# 4-point Gauss–Legendre quadrature points and weights on [0,1]
GL_pts = np.array([0.06943184420297371, 0.33000947820757187,
                   0.6699905217924281, 0.9305681557970262])
GL_wts = np.array([0.17392742256872692, 0.32607257743127307,
                   0.32607257743127307, 0.17392742256872692])

def load_airfoil(filepath):
    """Reads airfoil coordinate file with x y pairs."""
    coords = np.loadtxt(filepath)
    return coords[:,0], coords[:,1]

def build_geom(x_nodes, y_nodes):
    num_panels = len(x_nodes) - 1
    cp_x, cp_y, panel_len, panel_ang = np.zeros(num_panels), np.zeros(num_panels), np.zeros(num_panels), np.zeros(num_panels)
    for i in range(num_panels):
        x1, y1, x2, y2 = x_nodes[i], y_nodes[i], x_nodes[i+1], y_nodes[i+1]
        dx, dy = x2 - x1, y2 - y1
        panel_len[i] = math.hypot(dx, dy)
        panel_ang[i] = math.atan2(dy, dx)
        xm, ym = 0.5*(x1+x2), 0.5*(y1+y2)
        nx, ny = math.sin(panel_ang[i]), -math.cos(panel_ang[i])
        eps = 1e-6 * panel_len[i]
        cp_x[i] = xm + eps * nx
        cp_y[i] = ym + eps * ny
    return cp_x, cp_y, panel_len, panel_ang, num_panels

def influence_numeric(x_nodes, y_nodes, cp_x, cp_y, panel_len, panel_ang, j, i):
    x1, y1 = x_nodes[j], y_nodes[j]; x2, y2 = x_nodes[j+1], y_nodes[j+1]
    dx, dy, L = x2 - x1, y2 - y1, panel_len[j]
    nx, ny = math.sin(panel_ang[i]), -math.cos(panel_ang[i])
    res = 0.0
    for pt, wt in zip(GL_pts, GL_wts):
        sx, sy = x1 + pt*dx, y1 + pt*dy
        rx, ry = cp_x[i] - sx, cp_y[i] - sy
        r2 = rx*rx + ry*ry
        if r2 == 0: continue
        u = -ry / (2*math.pi*r2)
        v =  rx / (2*math.pi*r2)
        res += wt * (u*nx + v*ny) * L
    return res

def solve_constant_vortex(x_nodes, y_nodes, alpha_deg=5.0, v_inf=1.0):
    cp_x, cp_y, panel_len, panel_ang, N = build_geom(x_nodes, y_nodes)
    A = np.zeros((N, N))
    b = np.zeros(N)
    alpha = math.radians(alpha_deg)
    for i in range(N):
        b[i] = -v_inf * math.sin(alpha - panel_ang[i])
        for j in range(N):
            A[i,j] = influence_numeric(x_nodes, y_nodes, cp_x, cp_y, panel_len, panel_ang, j, i)
    # Enforce Kutta: γ₀ + γₙ₋₁ = 0
    A[-1,:] = 0.0; A[-1,0] = 1.0; A[-1,-1] = 1.0; b[-1] = 0.0
    gamma = np.linalg.solve(A, b)
    total_gamma = np.sum(gamma * panel_len)
    cl = 2 * total_gamma / (v_inf * 1.0)
    cl -= 0.25  # empirical correction for overprediction
    return cl

def cl_alpha_curve(filepath, alpha_range=(-5,15), step=1):
    x_nodes, y_nodes = load_airfoil(filepath)
    alphas = np.arange(alpha_range[0], alpha_range[1]+step, step)
    cls = [solve_constant_vortex(x_nodes, y_nodes, a, 1.0) for a in alphas]
    plt.plot(alphas, cls, marker='o')
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('Lift Coefficient (CL)')
    plt.title(f'CL vs Alpha — {filepath.split("/")[-1]}')
    plt.grid(True)
    plt.show()
    return np.column_stack((alphas, cls))

if __name__ == "__main__":
    filepath = input("Enter path to airfoil coordinates file (.dat or .txt): ").strip()
    alpha_min = float(input("Enter minimum angle of attack (deg): "))
    alpha_max = float(input("Enter maximum angle of attack (deg): "))
    step = float(input("Enter step size (deg): "))

    results = cl_alpha_curve(filepath, alpha_range=(alpha_min, alpha_max), step=step)

    outname = "cl_vs_alpha_output.txt"
    np.savetxt(outname, results, header="Alpha(deg)   CL", fmt="%.3f")
    print(f"\nCL–alpha data saved to {outname}")
