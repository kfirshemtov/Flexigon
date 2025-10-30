
def replace_vertices(in_path, out_path, v_hr_new):
    """
    Replace vertex positions in an OBJ file with new vertex coordinates.

    Args:
        in_path (str): Path to the input OBJ file.
        out_path (str): Path to the output OBJ file where the modified data will be saved.
        v_hr_new (np.ndarray or list of tuples): New vertex coordinates of shape (N, 3).

    This function replaces all lines starting with 'v ' in the input OBJ file
    with new vertex positions from v_hr_new and writes the result to out_path.
    """
    
    new_lines = []
    v_index = 0

    with open(in_path, 'r') as f:
        lines = f.readlines()

    # Replace only lines starting with "v "
    for line in lines:
        if line.startswith('v '):
            x, y, z = v_hr_new[v_index]
            new_line = f"v {x:.6f} {y:.6f} {z:.6f}\n"
            new_lines.append(new_line)
            v_index += 1
        else:
            new_lines.append(line)

    # Save to new file
    with open(out_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Replaced {v_index} vertices and wrote to {out_path}")

# Usage:
# replace_vertices(in_path, out_path, v_hr_new)
