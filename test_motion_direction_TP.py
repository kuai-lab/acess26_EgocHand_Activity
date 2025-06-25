import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import re
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def track_hand_path_from_folder(input_folder: str, output_video_path: str, track_hand: str = 'right', fps: int = 30):
    print(f"WiLoR 파이프라인 초기화: '{track_hand}' 손 추적 중...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    print("WiLoR 파이프라인 준비 완료.")

    image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')), key=natural_sort_key)
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)

    if not image_files:
        print(f"⚠️ 이미지 파일 없음: {input_folder}")
        return

    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    hand_path_points = []

    print(f"총 {len(image_files)}개 프레임 처리 시작...")
    for img_path in tqdm(image_files, desc="이미지 처리"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        outputs = pipe.predict(frame)

        target_hand_output = None
        if outputs:
            for out in outputs:
                if track_hand == 'right' and out['is_right'] == 1:
                    target_hand_output = out
                    break
                elif track_hand == 'left' and out['is_right'] == 0:
                    target_hand_output = out
                    break

        if target_hand_output:
            wrist_keypoint_2d = target_hand_output["wilor_preds"]["pred_keypoints_2d"][0][0]
            center_x, center_y = int(wrist_keypoint_2d[0]), int(wrist_keypoint_2d[1])
            hand_path_points.append((center_x, center_y))

        if len(hand_path_points) > 1:
            for i in range(1, len(hand_path_points)):
                cv2.line(frame, hand_path_points[i-1], hand_path_points[i], (0, 255, 0), thickness=3)

        if hand_path_points:
            cv2.circle(frame, hand_path_points[-1], radius=8, color=(0, 0, 255), thickness=-1)

        vout.write(frame)

    vout.release()
    print(f"✅ 저장 완료: {output_video_path}")


if __name__ == '__main__':
    base_dir = Path("/data/jyseo/data_MHAV/screw")
    output_base = Path("/data/jyseo/functional_hand_type/direction_hist_results/videos/screw")

    for subj_dir in base_dir.glob("*"):  # 예: hg, jl, sp ...
        if not subj_dir.is_dir():
            continue
        for seq_dir in subj_dir.glob("*"):
            rgb_dir = seq_dir / "RGB_undistorted"
            if not rgb_dir.exists():
                continue

            relative_path = rgb_dir.relative_to(base_dir)
            output_video_path = output_base / relative_path.parent / f"{relative_path.parent.name}_hand_path.mp4"
            output_video_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n🚀 시퀀스 처리 시작: {rgb_dir}")
            print(f"👉 출력 파일: {output_video_path}")

            track_hand_path_from_folder(
                input_folder=str(rgb_dir),
                output_video_path=str(output_video_path),
                fps=10
            )
