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
    Calculates the coefficient of lift (cl) and coefficient of moment
    about the quarter-chord (cm_c4) for an airfoil using Thin Airfoil Theory.
    """
    
    # Get the camber slope interpolator
    dz_dx_interp = _get_dz_dx_interpolator(top_coords, bottom_coords)
    
    # --- Step 3: Define Integrands for A0, A1, and A2 coefficients ---
    alpha_rad = np.radians(alpha_deg)
    
    def integrand_A0(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x == 0: x = 1e-6
        return dz_dx_interp(x)
    
    def integrand_A1(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x == 0: x = 1e-6
        return dz_dx_interp(x) * np.cos(theta)
    
    # NEW: Integrand for A2 coefficient
    def integrand_A2(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x == 0: x = 1e-6
        return dz_dx_interp(x) * np.cos(2 * theta)

    # --- Step 4: Integrate to find A0, A1, and A2 ---
    integral_A0_val, err_A0 = quad(integrand_A0, 0, np.pi)
    integral_A1_val, err_A1 = quad(integrand_A1, 0, np.pi)
    integral_A2_val, err_A2 = quad(integrand_A2, 0, np.pi) # NEW
    
    # --- Step 5: Calculate A0, A1, and A2 based on TAT formulas ---
    A0 = alpha_rad - (1 / np.pi) * integral_A0_val
    A1 = (2 / np.pi) * integral_A1_val
    A2 = (2 / np.pi) * integral_A2_val # NEW
    
    # --- Step 6: Calculate Coefficient of Lift (cl) ---
    cl = np.pi * (2 * A0 + A1)
    
    # --- Step 7: Calculate Coefficient of Moment (cm_c4) ---
    # NEW: Based on the formula Cm_c/4 = (pi/4)*(A2 - A1)
    cm_c4 = (np.pi / 4) * (A2 - A1)
    
    return cl, cm_c4

def calculate_zero_lift_angle(top_coords, bottom_coords):
    """
    Calculates the zero-lift angle of attack (alpha_l0) in degrees
    using the formula:
    alpha(l=0) = -1/pi * integral[0 to pi]( (dz/dx) * (cos(theta) - 1) ) d(theta)
    """
    
    # Get the camber slope interpolator
    dz_dx_interp = _get_dz_dx_interpolator(top_coords, bottom_coords)

    # --- Define Integrand for alpha_l0 ---
    def integrand_alpha_l0(theta):
        x = 0.5 * (1 - np.cos(theta))
        if x == 0: x = 1e-6
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
    print("Please provide paths to your coordinate files.")
    print("Files should be text files with two columns (X and Y), separated by spaces.")
    
    top_file_path = input("Enter the file path for the TOP surface coordinates: ")
    bottom_file_path = input("Enter the file path for the BOTTOM surface coordinates: ")

    # --- 2. Load Airfoil Data from Files ---
    try:
        # Use numpy.loadtxt to read the text files
        # This assumes X is in the first column (0) and Y in the second (1)
        top_coords = np.loadtxt(top_file_path)
        bottom_coords = np.loadtxt(bottom_file_path)
        
        print("\n...Files loaded successfully.")
        
    except FileNotFoundError:
        print(f"\n*** ERROR: File not found. ***")
        print(f"One of the paths was incorrect:")
        print(f"Top file: '{top_file_path}'")
        print(f"Bottom file: '{bottom_file_path}'")
        sys.exit("Exiting program.")
    except Exception as e:
        print(f"\n*** ERROR: Could not read files. ***")
        print(f"Please ensure they are simple text files with X Y columns.")
        print(f"Details: {e}")
        sys.exit("Exiting program.")

    
    # --- 3. Calculate and Print Zero-Lift Angle ---
    
    alpha_l0_custom = calculate_zero_lift_angle(top_coords, bottom_coords)
    
    print("-" * 50)
    print("Zero-Lift Angle of Attack (α_L=0) Calculation")
    print(f"  Your Airfoil: {alpha_l0_custom:.4f} degrees")
    print("-" * 50)

    # --- 4. Set up the Loop ---
    
    # Define the range of angles of attack to test
    alpha_range_deg = np.linspace(-5, 10, 16)  # From -5 to 10 degrees (16 steps)
    
    # Create empty lists to store the results
    cl_custom_list = []
    cm_custom_list = [] # NEW
    
    print("\nCalculating cl and cm vs. alpha curves...")
    
    # --- 5. Run the Loop ---
    for alpha in alpha_range_deg:
        # Calculate cl and cm for the custom airfoil
        cl_val, cm_val = calculate_aero_coefficients(top_coords, bottom_coords, alpha)
        cl_custom_list.append(cl_val)
        cm_custom_list.append(cm_val) # NEW
        
    print("...Calculations complete. Plotting results.")
    
    # --- 6. Plot the Results (NEW Twin-Axis Plot) ---
    
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # --- Plot 1: Cl vs. Alpha (Left Y-Axis) ---
    color_cl = 'blue'
    ax1.set_xlabel('Angle of Attack $\\alpha$ (degrees)')
    ax1.set_ylabel('Coefficient of Lift ($c_l$)', color=color_cl)
    ax1.plot(alpha_range_deg, cl_custom_list, color=color_cl, marker='o', 
             label=f'Your Airfoil $c_l$ (α_L=0 = {alpha_l0_custom:.2f}°)')
    ax1.tick_params(axis='y', labelcolor=color_cl)
    ax1.grid(True, linestyle='--') # Main grid
    
    # --- Plot 2: Cm vs. Alpha (Right Y-Axis) ---
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color_cm = 'red'
    ax2.set_ylabel('Coefficient of Moment ($c_{m,c/4}$)', color=color_cm)
    ax2.plot(alpha_range_deg, cm_custom_list, color=color_cm, marker='s', 
             linestyle='--', label='Your Airfoil $c_{m,c/4}$')
    ax2.tick_params(axis='y', labelcolor=color_cm)

    # --- 7. Add Plot Title and Combined Legend ---
    plt.title('Thin Airfoil Theory: $c_l$ and $c_{m,c/4}$ vs. $\\alpha$')
    
    # Get handles and labels from both axes to combine them
    lines_cl, labels_cl = ax1.get_legend_handles_labels()
    lines_cm, labels_cm = ax2.get_legend_handles_labels()
    ax1.legend(lines_cl + lines_cm, labels_cl + labels_cm, loc='upper left')

    # Add zero lines
    ax1.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax1.axvline(0, color='black', linestyle=':', linewidth=0.5)
    
    fig.tight_layout() # Adjust plot to prevent label overlap
    plt.show()