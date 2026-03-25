import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyRotation


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', str(s))]

def load_keypoints_from_csv_folder(folder_path):
    csv_files = sorted(Path(folder_path).glob("keypoint_*.csv"), key=natural_sort_key)
    keypoints_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.shape[0] != 42: continue
        keypoints = df[['x', 'y', 'z']].values.astype(np.float32)
        keypoints_list.append(keypoints)
    return keypoints_list

def extract_rotation_feature_from_keypoints(
    keypoints_list: list,
    track_hand: str = 'right',
    num_frames: int = 500,
    smoothing_window_size: int = 5
):
    valid_keypoints = []
    for kps in keypoints_list:
        right_hand_kps = kps[:21, :]
        left_hand_kps = kps[21:, :]
        is_right_detected = not np.all(right_hand_kps == 0)
        is_left_detected = not np.all(left_hand_kps == 0)
        if is_right_detected and is_left_detected:
            valid_keypoints.append(right_hand_kps if track_hand == 'right' else left_hand_kps)

    if not valid_keypoints:
        tqdm.write(f"Warning: 유효한 양손 키포인트를 찾지 못했습니다. 0 벡터를 반환합니다.")
        return np.zeros(num_frames)

    total_valid_frames = len(valid_keypoints)
    keypoints_to_process = valid_keypoints if total_valid_frames <= num_frames else valid_keypoints[(total_valid_frames - num_frames) // 2:(total_valid_frames - num_frames) // 2 + num_frames]

    last_grasp_points = None
    angle_history = []
    signed_accumulated_angle_deg = 0.0
    for keypoints_3d in keypoints_to_process:
        palm_center = (keypoints_3d[0] + keypoints_3d[5] + keypoints_3d[17]) / 3
        finger_center = (keypoints_3d[4] + keypoints_3d[8] + keypoints_3d[12]) / 3
        palm_to_finger_vector = finger_center - palm_center
        current_grasp_points = np.array([keypoints_3d[4], keypoints_3d[8], keypoints_3d[12]])
        if last_grasp_points is not None:
            delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
            rotvec = delta_rotation.as_rotvec()
            angle_increment_rad = np.linalg.norm(rotvec)
            sign = np.sign(np.dot(rotvec, palm_to_finger_vector)) if angle_increment_rad > 1e-6 else 0
            signed_delta_angle_deg = np.rad2deg(angle_increment_rad) * sign
            signed_accumulated_angle_deg += signed_delta_angle_deg
        angle_history.append(signed_accumulated_angle_deg)
        last_grasp_points = current_grasp_points.copy()

    feature_vector = np.array(angle_history)
    if len(feature_vector) < num_frames:
        padding = np.full(num_frames - len(feature_vector), feature_vector[-1] if len(feature_vector) > 0 else 0)
        feature_vector = np.concatenate([feature_vector, padding])

    if smoothing_window_size > 1:
        feature_vector = pd.Series(feature_vector).rolling(window=smoothing_window_size, min_periods=1, center=True).mean().to_numpy()

    return feature_vector

def process_csv_based_pose():
    base_input_folder = "../data_MHAV"
    track_hand = 'right'
    num_frames = 500
    smoothing_window = 5

    keypoint_folders = list(Path(base_input_folder).glob("**/wilor_pose_3d/processed_270_480/keypoint"))
    print(f"🔍 총 {len(keypoint_folders)}개의 keypoint 폴더 처리 시작\n")

    for folder in tqdm(keypoint_folders, desc="전체 진행"):
        action_name = f"{folder.parent.parent.parent.name}"
        keypoints_list = load_keypoints_from_csv_folder(folder)
        if not keypoints_list:
            tqdm.write(f"[WARN] {folder}: 키포인트 없음")
            continue

        rotation_feature = extract_rotation_feature_from_keypoints(
            keypoints_list=keypoints_list,
            track_hand=track_hand,
            num_frames=num_frames,
            smoothing_window_size=smoothing_window
        )
        output_path = folder.parent.parent.parent / f"{action_name}.npy"
        np.save(output_path, rotation_feature)
        tqdm.write(f"저장 완료: {output_path}")

    print("\n🎉 모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    process_csv_based_pose()
