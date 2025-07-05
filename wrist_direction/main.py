import os
import sys
sys.path.append(os.path.dirname(__file__))

from data_loader import load_wrist_trajectory_from_csv_folder
from feature_extractor import compute_rotation_direction_cross  # ✅ 새 함수 import
import numpy as np

def main():
    base_folder = "/data/jyseo/data_MHAV/unscrew/kl/unscrew_woodChunk-socketCapScrew_allenKey_kl/wilor_pose_3d/processed_270_480/keypoint"
    
    print(f"[INFO] 분석 대상 폴더: {base_folder}")

    try:
        traj3d = load_wrist_trajectory_from_csv_folder(base_folder)
        if traj3d.shape[0] < 3:
            print("[WARN] trajectory 길이가 너무 짧음. 스킵합니다.")
            return

        traj2d = traj3d[:, :2]  # XY 평면 투영
        direction, total_cross = compute_rotation_direction_cross(traj2d)
        print(f"{base_folder} → {direction} (cross sum: {total_cross:.4f})")
    except Exception as e:
        print(f"[ERROR] 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
