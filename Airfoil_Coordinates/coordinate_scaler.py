import os

def is_number(s):
    """Helper to check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    print("--- Airfoil Coordinate Scaler ---")
    
    # 1. Get the file path
    # .strip('"') removes quotes if you drag-and-drop the file into the terminal
    file_path = input("Enter the path to the coordinate file: ").strip('"')
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 2. Get the scale factor
    try:
        scale_input = input("Enter scaling factor (e.g., 1000 for m to mm, 25.4 for in to mm): ")
        scale_factor = float(scale_input)
    except ValueError:
        print("Error: Please enter a valid number for the scale.")
        return

    # 3. Create Output Filename
    # Splits "airfoil.dat" into "airfoil" and ".dat"
    base_name, ext = os.path.splitext(file_path)
    output_path = f"{base_name}_{scale_input}{ext}"

    try:
        with open(file_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                # Strip whitespace to check content, but keep original line for headers
                stripped = line.strip()
                
                # If line is empty, preserve it
                if not stripped:
                    f_out.write(line)
                    continue
                
                parts = stripped.split()
                
                # Check if the line is coordinate data (all parts are numbers)
                if all(is_number(p) for p in parts):
                    # Perform the scaling
                    # We create a new list of scaled values
                    scaled_values = [float(p) * scale_factor for p in parts]
                    
                    # Format preservation:
                    # We use {0:.6f} to keep 6 decimal places (standard for CAD imports)
                    # We join with a tab character to ensure distinct columns
                    new_line = "\t".join([f"{val:.6f}" for val in scaled_values]) + "\n"
                    f_out.write(new_line)
                else:
                    # This is likely a header or a Group definition (common in SpaceClaim)
                    # Write it exactly as it was in the original file
                    f_out.write(line)

        print(f"\nSuccess! File created: {output_path}")
        print(f"Original: {file_path}")
        print(f"Scaled by: {scale_factor}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()