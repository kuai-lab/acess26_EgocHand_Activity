import os
import pandas as pd
import numpy as np
from glob import glob

BASE_DIR = "/home/jyseo/hand_journal/data_MHAV/assemble/*"
target_folders = sorted(glob(os.path.join(BASE_DIR, "assemble_*_*_*")))

def extract_right_hand(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 42:
        return df.iloc[:21][["x", "y", "z"]].values
    return None

def rotation_vector(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot)
    axis = np.cross(v1, v2)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.zeros(3)
    axis = axis / norm
    return angle * axis

THRESHOLD = 0.2  # radian

print("\n================== 🧾 시퀀스별 회전 방향 통계 ==================\n")

for folder in target_folders:
    pose_folder = os.path.join(folder, "pose_2d_mp_csv")
    if not os.path.isdir(pose_folder):
        continue

    csv_files = sorted(glob(os.path.join(pose_folder, "*.csv")))
    if len(csv_files) < 2:
        continue

    count_twist = 0      # 🔂
    count_untwist = 0    # 🔁
    count_none = 0       # ⏸

    for i in range(len(csv_files) - 1):
        f1, f2 = csv_files[i], csv_files[i + 1]
        kp1 = extract_right_hand(f1)
        kp2 = extract_right_hand(f2)

        if kp1 is None or kp2 is None:
            continue

        palm1 = kp1[17] - kp1[1]  # 손 측면 회전에 민감한 축
        palm2 = kp2[17] - kp2[1]
        rot_vec = rotation_vector(palm1, palm2)

        rot_z = rot_vec[2]
        rot_norm = np.linalg.norm(rot_vec)

        if rot_norm < THRESHOLD:
            count_none += 1
        elif rot_z > 0:
            count_untwist += 1
        elif rot_z < 0:
            count_twist += 1
        else:
            count_none += 1

    print(f"📂 시퀀스: {os.path.basename(folder)}")
    print(f"🔂 조임 방향 : {count_twist}개")
    print(f"⏸ 거의 없음 : {count_none}개")
    print(f"🔁 풀림 방향 : {count_untwist}개\n")

print("===============================================================")
