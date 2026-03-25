import os
import pandas as pd
import numpy as np

def load_wrist_trajectory_from_csv_folder(folder_path):
    """
    folder_path: keypoint_*.csv 파일들이 있는 폴더 (프레임 단위)
    return: (T, 3) numpy array (손목 궤적)
    """
    file_list = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith("keypoint_") and f.endswith(".csv")
    ], key=lambda x: int(x.split("_")[-1].split(".")[0]))
    print(f"[DEBUG] Found {len(file_list)} keypoint CSVs in {folder_path}")

    trajectory = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        if len(df) == 0:
            continue
        wrist = df.iloc[0][['x', 'y', 'z']].values.astype(np.float32)
        trajectory.append(wrist)

    if len(trajectory) == 0:
        raise ValueError("No valid wrist keypoints found.")

    return np.stack(trajectory)  # [T, 3]
