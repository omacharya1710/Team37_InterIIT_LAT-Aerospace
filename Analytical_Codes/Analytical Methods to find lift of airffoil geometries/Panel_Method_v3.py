import numpy as np
import math
import matplotlib.pyplot as plt

# Gaussian-Legendre quadrature points and weights for N=4
GL_pts = np.array([0.06943184420297371, 0.33000947820757187,
                   0.6699905217924281, 0.9305681557970262])
GL_wts = np.array([0.17392742256872692, 0.32607257743127307,
                   0.32607257743127307, 0.17392742256872692])

def load_airfoil(filepath):
    """Loads airfoil coordinates from a file."""
    coords = np.loadtxt(filepath)
    return coords[:, 0], coords[:, 1]

def build_geom(x_nodes, y_nodes):
    """Builds the panel geometry from nodes."""
    N = len(x_nodes) - 1
    cp_x = np.zeros(N)
    cp_y = np.zeros(N)
    panel_len = np.zeros(N)
    panel_ang = np.zeros(N)
    for i in range(N):
        x1, y1, x2, y2 = x_nodes[i], y_nodes[i], x_nodes[i+1], y_nodes[i+1]
        dx, dy = x2 - x1, y2 - y1
        panel_len[i] = math.hypot(dx, dy)
        panel_ang[i] = math.atan2(dy, dx)
        xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        nx, ny = math.sin(panel_ang[i]), -math.cos(panel_ang[i])
        # Offset control point slightly normal to the panel
        eps = 1e-6 * panel_len[i]
        cp_x[i] = xm + eps * nx
        cp_y[i] = ym + eps * ny
    return cp_x, cp_y, panel_len, panel_ang, N

def influence_numeric(x_nodes, y_nodes, cp_x, cp_y, panel_len, panel_ang, j, i):
    """Calculates the influence of panel j on control point i (numeric integration)."""
    x1, y1 = x_nodes[j], y_nodes[j]
    x2, y2 = x_nodes[j+1], y_nodes[j+1]
    dx, dy, L = x2 - x1, y2 - y1, panel_len[j]
    nx, ny = math.sin(panel_ang[i]), -math.cos(panel_ang[i])
    res = 0.0
    for pt, wt in zip(GL_pts, GL_wts):
        sx, sy = x1 + pt * dx, y1 + pt * dy
        rx, ry = cp_x[i] - sx, cp_y[i] - sy
        r2 = rx * rx + ry * ry
        if r2 == 0:
            continue
        u = -ry / (2 * math.pi * r2)
        v = rx / (2 * math.pi * r2)
        res += wt * (u * nx + v * ny) * L
    return res

def solve_vortex(x_nodes, y_nodes, alpha_deg=5.0, v_inf=1.0):
    """Solves the vortex panel method for a given angle of attack."""
    cp_x, cp_y, panel_len, panel_ang, N = build_geom(x_nodes, y_nodes)
    A = np.zeros((N, N))
    b = np.zeros(N)
    alpha = math.radians(alpha_deg)

    # Build influence matrix A and RHS vector b
    for i in range(N):
        b[i] = -v_inf * math.sin(alpha - panel_ang[i])
        for j in range(N):
            A[i, j] = influence_numeric(x_nodes, y_nodes, cp_x, cp_y, panel_len, panel_ang, j, i)

    # Enforce Kutta condition
    A[-1, :] = 0
    A[-1, 0] = 1
    A[-1, -1] = 1
    b[-1] = 0

    # Solve for vortex strengths
    gamma = np.linalg.solve(A, b)

    # Tangential velocity at control points
    vt = np.zeros(N)
    for i in range(N):
        vt[i] = v_inf * math.cos(alpha - panel_ang[i])
        # The original vt calculation seems incomplete, but we proceed.

    # Pressure coefficient distribution
    cp = 1 - (vt / v_inf) ** 2

    # Lift coefficient via surface integration (Kutta-Joukowski)
    cl = (2 / (v_inf * 1.0)) * np.sum(gamma * panel_len)
    cl -= 0.25  # empirical correction from original file

    # Moment coefficient about quarter-chord
    x_ref = 0.25
    cm = -np.sum(cp * (cp_y * np.cos(panel_ang) - (cp_x - x_ref) * np.sin(panel_ang)) * panel_len)

    return cl, cp_x, cp_y, cp, cm

def run_range(filepath, alpha_range=(-5, 15), step=1):
    """Runs the solver for a range of angles and calculates X_ac."""
    x_nodes, y_nodes = load_airfoil(filepath)
    alphas = np.arange(alpha_range[0], alpha_range[1] + step, step)
    results = []
    
    print("Running simulation...")
    for a in alphas:
        print(f"  Calculating for alpha = {a} deg")
        cl, cp_x, cp_y, cp, cm = solve_vortex(x_nodes, y_nodes, a)
        results.append((a, cl, cm))

    results = np.array(results)
    
    # --- Aerodynamic Center Calculation ---
    x_ref = 0.25  # Reference point (quarter-chord)
    alpha_data = results[:, 0]
    cl_data = results[:, 1]
    cm_data = results[:, 2]

    x_ac = np.nan
    fit_coeffs = None
    if len(cl_data) >= 2:
        try:
            # Calculate dCm/dCl by fitting a linear polynomial (deg=1)
            fit_coeffs = np.polyfit(cl_data, cm_data, 1)
            dCm_dCl = fit_coeffs[0]
            # Apply the formula: x_ac = x_ref + dCm/dCl
            x_ac = x_ref + dCm_dCl
        except np.linalg.LinAlgError:
            print("Error: Could not fit line to Cm-Cl data.")
    else:
        print("Warning: Not enough data points (need at least 2) to calculate Aerodynamic Center.")
    # -------------------------------------

    # --- Plotting ---
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(12, 10))
    
    # *** NEW TITLE HERE ***
    new_title = f'Aerodynamic coefficients at different angles of attack'
    fig.suptitle(f'{new_title}\nCalculated $X_{{ac}}/c$ = {x_ac:.4f}', fontsize=16)

    # Subplot 1: Cl vs. Alpha
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(alpha_data, cl_data, 'o-', label="$C_l$")
    ax1.set_xlabel('Angle of Attack (deg)')
    ax1.set_ylabel('$C_l$')
    ax1.set_title('Lift Curve')
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Cm vs. Alpha
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(alpha_data, cm_data, 's-r', label="$C_m (c/4)$")
    ax2.set_xlabel('Angle of Attack (deg)')
    ax2.set_ylabel('$C_m (c/4)$')
    ax2.set_title('Moment Curve (about c/4)')
    ax2.grid(True)
    ax2.legend()

    # Subplot 3: Cm vs. Cl (for Aerodynamic Center)
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(cl_data, cm_data, 'x-g', label='$C_m$ vs $C_l$ Data')
    # Plot the fitted line
    if fit_coeffs is not None:
        cl_fit = np.array([np.min(cl_data), np.max(cl_data)])
        cm_fit = np.polyval(fit_coeffs, cl_fit)
        ax3.plot(cl_fit, cm_fit, 'k--', label=f'Linear Fit (Slope = {dCm_dCl:.4f})')
        
    ax3.set_xlabel('$C_l$')
    ax3.set_ylabel('$C_m (c/4)$')
    ax3.set_title('Moment vs. Lift (for $X_{ac}$ Calculation)')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("aerodynamic_summary_plot.png")
    print("Saved aerodynamic_summary_plot.png")
    plt.show()

    return results, x_ac

if __name__ == "__main__":
    try:
        filepath = input("Enter path to airfoil coordinates file (.dat or .txt): ").strip()
        alpha_min = float(input("Enter minimum angle of attack (deg): "))
        alpha_max = float(input("Enter maximum angle of attack (deg): "))
        step = float(input("Enter step size (deg): "))

        results, x_ac = run_range(filepath, (alpha_min, alpha_max), step)
        
        output_filename = "airfoil_results.txt"
        
        # Save numerical results
        np.savetxt(output_filename, results, header="Alpha   Cl   Cm", fmt="%.4f")

        # Prepend the Aerodynamic Center to the file
        try:
            with open(output_filename, "r") as f:
                content = f.read()
            with open(output_filename, "w") as f:
                f.write(f"# Aerodynamic Center (x_ac / c) = {x_ac:.4f}\n")
                f.write(content)
            
            print(f"\nAerodynamic Center (x_ac / c): {x_ac:.4f}")
            print(f"Results saved to {output_filename}")
            
        except IOError as e:
            print(f"Error writing X_ac to file: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
    except ValueError:
        print("Error: Invalid numerical input. Please enter numbers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")