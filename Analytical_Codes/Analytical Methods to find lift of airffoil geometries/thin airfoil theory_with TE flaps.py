import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys  # Import sys to handle program exit

def _get_dz_dx_interpolator(top_coords, bottom_coords):
    """
    Helper function to process coordinates and return a
    dz/dx (camber line slope) interpolation function.
    """
    # --- Step 1: Process Coordinates and Calculate Camber Line ---
    top_coords = top_coords[top_coords[:, 0].argsort()]
    bottom_coords = bottom_coords[bottom_coords[:, 0].argsort()]
    
    interp_top = interp1d(top_coords[:, 0], top_coords[:, 1], 
                          kind='cubic', fill_value="extrapolate")
    interp_bottom = interp1d(bottom_coords[:, 0], bottom_coords[:, 1], 
                             kind='cubic', fill_value="extrapolate")
    
    x_camber = np.linspace(0.0001, 0.9999, 201) # Avoid exact 0/1 for stability
    
    y_top = interp_top(x_camber)
    y_bottom = interp_bottom(x_camber)
    z_camber = (y_top + y_bottom) / 2
    
    # --- Step 2: Calculate Camber Line Slope (dz/dx) ---
    dz_dx = np.gradient(z_camber, x_camber)
    dz_dx_interp = interp1d(x_camber, dz_dx, 
                            kind='cubic', fill_value="extrapolate")
    
    return dz_dx_interp

def calculate_aero_coefficients(top_coords, bottom_coords, alpha_deg):
    """
    Calculates the BASELINE coefficient of lift (cl) and moment (cm_c4)
    due to CAMBER and ANGLE OF ATTACK only.
    (Flap contributions will be added separately)
    """
    
    # Get the camber slope interpolator
    dz_dx_interp = _get_dz_dx_interpolator(top_coords, bottom_coords)
    
    # --- Step 3: Define Integrands for A0, A1, and A2 coefficients ---
    alpha_rad = np.radians(alpha_deg)
    
    def integrand_A0(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x <= 0: x = 1e-6
        if x >= 1: x = 0.9999
        return dz_dx_interp(x)
    
    def integrand_A1(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x <= 0: x = 1e-6
        if x >= 1: x = 0.9999
        return dz_dx_interp(x) * np.cos(theta)
    
    def integrand_A2(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x <= 0: x = 1e-6
        if x >= 1: x = 0.9999
        return dz_dx_interp(x) * np.cos(2 * theta)

    # --- Step 4: Integrate to find A0, A1, and A2 ---
    integral_A0_val, err_A0 = quad(integrand_A0, 0, np.pi)
    integral_A1_val, err_A1 = quad(integrand_A1, 0, np.pi)
    integral_A2_val, err_A2 = quad(integrand_A2, 0, np.pi) 
    
    # --- Step 5: Calculate A0, A1, and A2 based on TAT formulas ---
    A0 = alpha_rad - (1 / np.pi) * integral_A0_val
    A1 = (2 / np.pi) * integral_A1_val
    A2 = (2 / np.pi) * integral_A2_val 
    
    # --- Step 6: Calculate Coefficient of Lift (cl) ---
    cl = np.pi * (2 * A0 + A1)
    
    # --- Step 7: Calculate Coefficient of Moment (cm_c4) ---
    cm_c4 = (np.pi / 4) * (A2 - A1)
    
    return cl, cm_c4

def calculate_zero_lift_angle(top_coords, bottom_coords):
    """
    Calculates the BASELINE zero-lift angle of attack (alpha_l0) in degrees
    due to CAMBER ONLY.
    """
    
    # Get the camber slope interpolator
    dz_dx_interp = _get_dz_dx_interpolator(top_coords, bottom_coords)

    # --- Define Integrand for alpha_l0 ---
    def integrand_alpha_l0(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x <= 0: x = 1e-6
        if x >= 1: x = 0.9999
        # This is the (dz/dx) * (cos(theta) - 1) part
        return dz_dx_interp(x) * (np.cos(theta) - 1)

    # --- Integrate ---
    integral_val, err = quad(integrand_alpha_l0, 0, np.pi)

    # --- Calculate zero-lift angle (in radians) ---
    alpha_l0_rad = (-1 / np.pi) * integral_val
    
    # Convert to degrees and return
    return np.degrees(alpha_l0_rad)

# ===================================================================
# --- MAIN SCRIPT EXECUTION ---
# ===================================================================

if __name__ == "__main__":
    
    # --- 1. Get File Paths from User ---
    print("--- Airfoil File Input ---")
    print("Files should be text files with two columns (X and Y), separated by spaces.")
    
    top_file_path = input("Enter the file path for the TOP surface coordinates: ")
    bottom_file_path = input("Enter the file path for the BOTTOM surface coordinates: ")

    # --- 2. Load Airfoil Data from Files ---
    try:
        top_coords = np.loadtxt(top_file_path)
        bottom_coords = np.loadtxt(bottom_file_path)
        print("\n...Files loaded successfully.")
    except FileNotFoundError:
        print(f"\n*** ERROR: File not found. ***")
        sys.exit("Exiting program.")
    except Exception as e:
        print(f"\n*** ERROR: Could not read files. ***\nDetails: {e}")
        sys.exit("Exiting program.")

    
    # --- 3. Get Flap Inputs from User (MODIFIED) ---
    print("\n--- Flap Inputs ---")
    try:
        xf_c = float(input("Enter the flap hinge location (x_f/c), e.g., 0.75: "))
        if not (0 < xf_c < 1):
             print("Warning: Flap hinge location should be between 0 and 1.")

    except ValueError:
        print("\n*** ERROR: Invalid input. Please enter a number for hinge location. ***")
        sys.exit("Exiting program.")
        
    # --- SET DELTA VALUES (Hard-coded as requested) ---
    delta_deg_list = [-5, 0, 5] 
    delta_rad_list = [np.radians(d) for d in delta_deg_list]
    
    print(f"Using fixed flap deflections (delta): {delta_deg_list} degrees")
        

    # --- 4. Calculate Baseline (Camber) Aero Properties ---
    
    # This is the zero-lift angle from *camber alone*
    alpha_l0_camber_deg = calculate_zero_lift_angle(top_coords, bottom_coords)
    alpha_l0_camber_rad = np.radians(alpha_l0_camber_deg)
    
    print("-" * 50)
    print("Baseline (Camber-Only) Properties")
    print(f"  Zero-Lift Angle (α_L=0) from Camber: {alpha_l0_camber_deg:.4f} degrees")
    print("-" * 50)

    # --- 5. Calculate Flap Derivatives (NEW) ---
    # Based on Thin Airfoil Theory formulas for a plain flap
    
    theta_f = np.arccos(1 - 2 * xf_c)
    
    # Lift derivative w.r.t. flap deflection (Cl_delta)
    cl_delta = 2 * (np.pi - theta_f + np.sin(theta_f))
    
    # Moment derivative w.r.t. flap deflection (Cm_c/4_delta)
    cm_delta = -0.5 * (np.sin(theta_f) + np.sin(2 * theta_f))
    
    # Flap effectiveness (tau)
    flap_effectiveness = cl_delta / (2 * np.pi)

    print("Flap Hinge Properties (from TAT)")
    print(f"  Hinge Location (x_f/c): {xf_c}")
    print(f"  Hinge Angle (θ_f): {np.degrees(theta_f):.2f} degrees")
    print(f"  Lift Derivative (C_l_δ): {cl_delta:.4f} (per radian)")
    print(f"  Moment Derivative (C_m_δ): {cm_delta:.4f} (per radian)")
    print(f"  Flap Effectiveness (τ): {flap_effectiveness:.4f}")
    print("-" * 50)

    # --- 6. Set up the Loop & Plot ---
    
    alpha_range_deg = np.linspace(-10, 15, 26)  # From -10 to 15 degrees
    
    fig, ax1 = plt.subplots(figsize=(11, 8))
    ax2 = ax1.twinx()  # Create a second y-axis
    
    print("\nCalculating cl and cm curves for each flap deflection...")
    
    # --- 7. Run Loops (Outer: delta, Inner: alpha) (MODIFIED) ---
    
    for delta_deg, delta_rad in zip(delta_deg_list, delta_rad_list):
        
        cl_total_list = []
        cm_total_list = []
        
        # Calculate the new total alpha_L=0 for this delta
        # alpha_L=0_total = alpha_L=0_camber - (Cl_delta * delta) / (2*pi)
        alpha_l0_total_rad = alpha_l0_camber_rad - (cl_delta * delta_rad) / (2 * np.pi)
        alpha_l0_total_deg = np.degrees(alpha_l0_total_rad)
        
        print(f"  Processing delta = {delta_deg:<5} deg (new α_L=0 = {alpha_l0_total_deg:.2f} deg)")
        
        # Inner loop: iterate over all angles of attack
        for alpha_deg in alpha_range_deg:
            
            # 1. Get baseline cl/cm from camber + alpha
            cl_camber, cm_camber = calculate_aero_coefficients(top_coords, bottom_coords, alpha_deg)
            
            # 2. Add flap contribution
            # Cl_total = Cl_camber_and_alpha + Cl_delta * delta
            cl_total = cl_camber + (cl_delta * delta_rad)
            
            # Cm_total = Cm_camber + Cm_delta * delta
            cm_total = cm_camber + (cm_delta * delta_rad)
            
            cl_total_list.append(cl_total)
            cm_total_list.append(cm_total)
            
        # 3. Plot this flap's curves
        # Plot Cl on the left axis (ax1)
        p1 = ax1.plot(alpha_range_deg, cl_total_list, marker='o', markersize=4,
                      label=f'$c_l$ ($\delta$={delta_deg}°)')
        
        # Plot Cm on the right axis (ax2)
        # Get color from the Cl plot to match them
        line_color = p1[0].get_color() 
        ax2.plot(alpha_range_deg, cm_total_list, marker='s', markersize=4,
                 linestyle='--', color=line_color,
                 label=f'$c_m$ ($\delta$={delta_deg}°)')

    
    print("...Calculations complete. Plotting results.")
    
    # --- 8. Format and Show Plot (MODIFIED) ---
    
    # --- Axis 1: Cl vs. Alpha (Left Y-Axis) ---
    color_cl = 'blue'
    ax1.set_xlabel('Angle of Attack $\\alpha$ (degrees)')
    ax1.set_ylabel('Coefficient of Lift ($c_l$)')
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--') # Main grid
    
    # --- Axis 2: Cm vs. Alpha (Right Y-Axis) ---
    color_cm = 'red'
    ax2.set_ylabel('Coefficient of Moment ($c_{m,c/4}$)')
    ax2.tick_params(axis='y')

    # --- Title and Combined Legend ---
    plt.title(f'Thin Airfoil Theory: $c_l$ & $c_m$ vs. $\\alpha$ with Flap (x_f/c = {xf_c})')
    
    # Get handles and labels from both axes to combine them
    lines_cl, labels_cl = ax1.get_legend_handles_labels()
    lines_cm, labels_cm = ax2.get_legend_handles_labels()
    # Put all legends in one box
    ax1.legend(lines_cl + lines_cm, labels_cl + labels_cm, loc='best', fontsize='small')

    # Add zero lines
    ax1.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax1.axvline(0, color='black', linestyle=':', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle=':', linewidth=0.5)
    
    fig.tight_layout() # Adjust plot to prevent label overlap
    plt.show()