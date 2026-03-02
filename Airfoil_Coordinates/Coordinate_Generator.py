import numpy as np
import sys

def generate_naca4digit_coords(m, p, t, num_points=50):
    """
    Generates x, y coordinates for a NACA 4-digit airfoil.
    
    m : max camber (e.g., 0.04 for 4%)
    p : position of max camber (e.g., 0.4 for 40%)
    t : max thickness (e.g., 0.12 for 12%)
    num_points : number of points for EACH surface (top/bottom)
    """
    
    # Create a cosine-spaced x-coordinate array
    # This clusters points near the leading and trailing edges
    theta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(theta))
    
    # Thickness distribution (y_t)
    y_t = (t / 0.2) * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                       0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line (y_c) and camber slope (dy_c_dx)
    y_c = np.zeros_like(x)
    dy_c_dx = np.zeros_like(x)
    
    # Calculate camber and slope
    # We must check for p=0 (symmetric airfoil) to avoid division by zero
    if p == 0:
        # For m=0, p=0 (e.g., NACA 0012), y_c and dy_c_dx are already zeros
        pass
    else:
        for i, x_i in enumerate(x):
            if x_i < p:
                y_c[i] = m / p**2 * (2 * p * x_i - x_i**2)
                dy_c_dx[i] = 2 * m / p**2 * (p - x_i)
            else:
                y_c[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x_i - x_i**2)
                dy_c_dx[i] = 2 * m / (1 - p)**2 * (p - x_i)
            
    # Angle of the camber line slope
    theta_c = np.arctan(dy_c_dx)
    
    # Final coordinates
    x_top = x - y_t * np.sin(theta_c)
    y_top = y_c + y_t * np.cos(theta_c)
    
    x_bottom = x + y_t * np.sin(theta_c)
    y_bottom = y_c - y_t * np.cos(theta_c)
    
    # Ensure (1,0) is the last point and (0,0) is the first
    x_top[-1], y_top[-1] = 1.0, 0.0
    x_bottom[-1], y_bottom[-1] = 1.0, 0.0
    x_top[0], y_top[0] = 0.0, 0.0
    x_bottom[0], y_bottom[0] = 0.0, 0.0

    # Combine into (N, 2) arrays
    top_coords = np.vstack((x_top, y_top)).T
    bottom_coords = np.vstack((x_bottom, y_bottom)).T
    
    return top_coords, bottom_coords

# --- Main part of the script ---
if __name__ == "__main__":
    
    # 1. Get user input
    naca_str = input("Enter a 4-digit NACA airfoil number (e.g., 4412): ")
    
    # 2. Validate the input
    if not naca_str.isdigit() or len(naca_str) != 4:
        print("Error: Input must be exactly 4 digits.")
        sys.exit()
        
    # 3. Parse the NACA string
    m_digit = int(naca_str[0])
    p_digit = int(naca_str[1])
    t_digits = int(naca_str[2:])
    
    # Check for invalid combination
    # (Camber position 'p' can't be 0 if camber 'm' is not 0)
    if m_digit > 0 and p_digit == 0:
        print(f"Error: Invalid NACA designation '{naca_str}'.")
        print("The camber position (2nd digit) cannot be 0 if")
        print("the max camber (1st digit) is not 0.")
        sys.exit()

    # Convert to fractional values
    M = m_digit / 100.0  # Max camber
    P = p_digit / 10.0   # Camber position
    T = t_digits / 100.0 # Thickness
    
    print(f"\nGenerating NACA {naca_str} with:")
    print(f"  Max Camber (m): {M*100:.0f}%")
    print(f"  Camber Pos (p): {P*100:.0f}%")
    print(f"  Thickness (t):  {T*100:.0f}%")
    
    # 4. Generate the coordinates
    top_data, bottom_data = generate_naca4digit_coords(M, P, T)
    
    # 5. Define file names
    top_file = f"naca{naca_str}_top.txt"
    bottom_file = f"naca{naca_str}_bottom.txt"
    
    # 6. Save the files
    np.savetxt(top_file, top_data, fmt="%.8f")
    np.savetxt(bottom_file, bottom_data, fmt="%.8f")
    
    print(f"\nSuccessfully created:")
    print(f"  - {top_file}")
    print(f"  - {bottom_file}")