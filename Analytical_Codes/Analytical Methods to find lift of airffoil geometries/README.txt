# Dependencies
All scripts require the following Python libraries:
- numpy
- matplotlib
- scipy (only for the thin airfoil theory scripts)

Install them using:
pip install numpy matplotlib scipy

#1. Panel_Method_v3.py

# Purpose
Implements a vortex panel method to compute lift coefficient (Cₗ), moment coefficient (Cₘ), and aerodynamic center (X_ac) of an airfoil.

# Inputs
- Airfoil coordinates file (.dat or .txt) with two columns (x, y).
- Angle of attack range:
  - Minimum angle (°)
  - Maximum angle (°)
  - Step size (°)

# Outputs
- airfoil_results.txt — text file containing α, Cₗ, and Cₘ.
- aerodynamic_summary_plot.png — plot of:
  - Cₗ vs. α
  - Cₘ vs. α
  - Cₘ vs. Cₗ (used to compute X_ac)

# Example Usage
python Panel_Method_v3.py
Example Input:
Enter path to airfoil coordinates file (.dat or .txt): NACA2412.txt
Enter minimum angle of attack (deg): -5
Enter maximum angle of attack (deg): 15
Enter step size (deg): 2

Output Files:
- airfoil_results.txt
- aerodynamic_summary_plot.png

---

# 2. thin airfoil theory.py

# Purpose
Computes lift and moment coefficients for a cambered airfoil using Thin Airfoil Theory, and determines the zero-lift angle of attack (α_L=0).

# Inputs
- Top surface coordinates file (X, Y)
- Bottom surface coordinates file (X, Y)

# Outputs
- Printed zero-lift angle α_L=0 (in degrees)
- Graph showing:
  - Cₗ vs. α (left y-axis)
  - Cₘ(c/4) vs. α (right y-axis)

# Example Usage
python "thin airfoil theory.py"
Example Input:
Enter the file path for the TOP surface coordinates: naca2412_top.txt
Enter the file path for the BOTTOM surface coordinates: naca2412_bottom.txt

---

# 3. thin airfoil theory_with TE flaps.py

# Purpose
Extends Thin Airfoil Theory to include plain trailing-edge flap effects on lift and moment coefficients.

# Inputs
- Top surface coordinates file (X, Y)
- Bottom surface coordinates file (X, Y)
- Flap hinge location (x_f / c)
- Fixed flap deflections δ = [-5°, 0°, +5°] (predefined)

# Outputs
- Printed baseline α_L=0 (camber only)
- Flap derivative parameters:
  - Cₗ_δ, Cₘ_δ, and flap effectiveness τ
- Graph showing Cₗ and Cₘ vs. α for each δ value

# Example Usage
python "thin airfoil theory_with TE flaps.py"
Example Input:
Enter the file path for the TOP surface coordinates: naca2412_top.txt
Enter the file path for the BOTTOM surface coordinates: naca2412_bottom.txt
Enter the flap hinge location (x_f/c), e.g., 0.75: 0.8

Output:
Interactive plot comparing lift and moment behavior for different flap deflections.
