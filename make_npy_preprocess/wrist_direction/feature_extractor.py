import numpy as np

def compute_signed_area_xy(points):
    """
    points: (T, 2) numpy array
    return: signed area (float)
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])



def compute_rotation_direction_cross(traj2d):
    """
    중심 기준 cross product 누적으로 CW/CCW 회전 방향을 판별
    """
    center = np.mean(traj2d, axis=0)
    vectors = traj2d - center

    total = 0.0
    for i in range(1, len(vectors)):
        v1 = vectors[i - 1]
        v2 = vectors[i]
        z_cross = np.cross(v1, v2)  # scalar 값 (z방향)
        total += z_cross

    direction = "CCW" if total > 0 else "CW"
    return direction, total
