"""
utils.py

general helper/utility functions
"""
import numpy as np

def find_elbow(x_vals, y_vals):
    """
    Find the elbow point of the x_vals vs y_vals graph; by the maximum
    perpendicular distance to the line joining first and last points.

    Args:
        x_vals(iter): iterable containing the x values
        y_vals(iter): iterable containing the y values

    Returns:
        elbow_x(float): The x-value at the elbow.
        elbow_y(float): The y-value at the elbow.
    """
    # Creating numpy arrays
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Line joining first and last points
    p_start = np.array([x[0], y[0]])
    p_end   = np.array([x[-1], y[-1]])
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)

    # Vector from start to each point
    vecs = np.column_stack((x - x[0], y - y[0]))

    # Obtaining the perpendicular distances using the cross product magnitude:
    cross_mag = np.abs(line_vec[0] * vecs[:,1] - line_vec[1] * vecs[:,0])
    distances = cross_mag / line_len

    # Elbow is the point with the maximum distance
    elbow_idx = int(np.argmax(distances))
    elbow_x   = x[elbow_idx].item()
    elbow_y   = y[elbow_idx].item()

    return elbow_x, elbow_y