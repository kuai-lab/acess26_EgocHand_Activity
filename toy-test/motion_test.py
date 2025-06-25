import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm

# 설정
pose_csv_folder = "/home/jyseo/hand_journal/data_MHAV/assemble/hg/assemble_hexNut-bigBolt3_hand_hg/pose_2d_mp_csv"
csv_files = sorted(glob(os.path.join(pose_csv_folder, "keypoint_*.csv")))
H, W = 480, 848
window_size = 10
stride = 5
output_dir = "motion_test_cleaned"
os.makedirs(output_dir, exist_ok=True)

def get_frame_index(filename):
    return int(os.path.basename(filename).split("_")[1].split(".")[0])

def get_color(t, T):
    t_norm = t / (T - 1)
    r = max(1 - 2 * t_norm, 0)
    g = 1 - abs(2 * t_norm - 1)
    b = max(2 * t_norm - 1, 0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR

# 오른손 필터링
filtered = []
# 개선된 필터링
for path in tqdm(csv_files, desc="📥 오른손 필터링"):
    df = pd.read_csv(path)
    if len(df) != 42:
        continue
    kps = df.iloc[21:42, [0, 1]].values.astype(np.float32)
    
    # 하나라도 0이면 NaN 처리
    kps[(kps[:, 0] == 0) | (kps[:, 1] == 0)] = np.nan
    if not np.isnan(kps).any():
        filtered.append((get_frame_index(path), path, kps))

# 연속된 segment 단위로 그룹핑
segments = []
current_segment = []
for i in range(len(filtered)):
    if not current_segment:
        current_segment.append(filtered[i])
    else:
        if filtered[i][0] == current_segment[-1][0] + 1:
            current_segment.append(filtered[i])
        else:
            if len(current_segment) >= window_size:
                segments.append(current_segment)
            current_segment = [filtered[i]]
if len(current_segment) >= window_size:
    segments.append(current_segment)

segment_idx = 0

# 슬라이딩 윈도우 시각화
for segment in segments:
    for i in range(0, len(segment) - window_size + 1, stride):
        frame_window = segment[i:i + window_size]
        keypoints_arr = np.array([f[2] for f in frame_window])  # (T, 21, 2)
        keypoints_arr = np.transpose(keypoints_arr, (1, 0, 2))  # (21, T, 2)
        T = keypoints_arr.shape[1]

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for joint_id in range(21):
            for t in range(1, T):
                pt1 = keypoints_arr[joint_id, t - 1]
                pt2 = keypoints_arr[joint_id, t]
                if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
                    continue
                color = get_color(t, T)
                cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, 1)

        out_path = os.path.join(output_dir, f"motion_segment_{segment_idx:03d}.png")
        cv2.imwrite(out_path, canvas)
        print(f"✅ 저장 완료: {out_path}")
        segment_idx += 1
