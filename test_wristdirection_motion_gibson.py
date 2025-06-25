## wilor mesh로 


import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm # 진행 상황을 보여주기 위한 라이브러리 (pip install tqdm)
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import re 

def natural_sort_key(s):
    """
    자연 정렬을 위한 키를 반환하는 함수.
    문자열에서 숫자 부분을 찾아 정수로 변환.
    예: "RGB_undistorted10.jpg" -> 10
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def track_hand_path_from_folder(input_folder: str, output_video_path: str, track_hand: str = 'right', fps: int = 30):
    """
    폴더 안의 이미지 시퀀스를 분석하여 손의 이동 경로를 비디오로 생성합니다.

    Args:
        input_folder (str): 연속된 이미지 파일들이 담긴 폴더 경로.
        output_video_path (str): 생성될 비디오 파일이 저장될 경로.
        fps (int): 출력 비디오의 초당 프레임 수.
    """
    print(f"WiLoR 파이프라인을 초기화하는 중입니다... '{track_hand}' 손을 추적합니다.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    print("파이프라인 초기화 완료.")

    # 입력 폴더에서 이미지 목록을 가져와 이름순으로 정렬
    image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')), key=natural_sort_key)
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)

    if not image_files:
        print(f"오류: '{input_folder}' 폴더에서 이미지 파일을 찾을 수 없습니다.")
        return

    # 첫 번째 이미지로 비디오의 크기를 결정
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # 비디오 작성을 위한 VideoWriter 객체 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 손의 중심점(손목 좌표)을 저장할 리스트
    hand_path_points = []
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
    for img_path in tqdm(image_files, desc="이미지 처리 중"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # wilor로 손 예측
        outputs = pipe.predict(frame)

        target_hand_output = None
        if outputs:
            # 감지된 모든 손을 확인
            for out in outputs:
                # is_right: 1이면 오른손, 0이면 왼손
                if track_hand == 'right' and out['is_right'] == 1:
                    target_hand_output = out
                    break # 원하는 손을 찾았으므로 반복 중단
                elif track_hand == 'left' and out['is_right'] == 0:
                    target_hand_output = out
                    break # 원하는 손을 찾았으므로 반복 중단
        
        # 원하는 손을 찾았을 경우에만 경로를 그림
        if target_hand_output:
            wrist_keypoint_2d = target_hand_output["wilor_preds"]["pred_keypoints_2d"][0][0]

            center_x, center_y = int(wrist_keypoint_2d[0]), int(wrist_keypoint_2d[1])
            hand_path_points.append((center_x, center_y))

            
        # 경로 그리기 (2개 이상의 점이 있을 때부터)
        if len(hand_path_points) > 1:
            for i in range(1, len(hand_path_points)):
                # 이전 점과 현재 점을 선으로 연결
                cv2.line(frame, hand_path_points[i-1], hand_path_points[i], (0, 255, 0), thickness=3)

        # 현재 손의 위치에 원으로 표시
        if hand_path_points:
            cv2.circle(frame, hand_path_points[-1], radius=8, color=(0, 0, 255), thickness=-1)

        # 처리된 프레임을 비디오에 추가
        vout.write(frame)

    # 모든 작업 완료 후 객체 해제
    vout.release()
    print("-" * 50)
    print(f"성공! 동영상 경로 추적이 완료되었습니다.")
    print(f"결과물이 '{output_video_path}'에 저장되었습니다.")


if __name__ == '__main__':
    # --- 사용자 설정 영역 ---
    
    # 1. 이미지가 들어있는 폴더 경로를 지정하세요.
    # 예: "C:/my_project/hand_images"
    # 폴더에 image_001.png, image_002.png, ... 와 같이 순서대로 정렬될 수 있는 파일들이 있어야 합니다.
    INPUT_IMAGE_FOLDER = "/data/jyseo/data_MHAV/screw/hg/screw_woodChunk-socketCapScrew_allenKey_hg/RGB_undistorted" 
    
    # 2. 결과 비디오를 저장할 경로와 파일명을 지정하세요.
    OUTPUT_VIDEO = "/data/jyseo/functional_hand_type/direction_hist_results/gibson/trajectory_results/wilor/test2.mp4"
    
    # --- 설정 영역 끝 ---
    
    # 결과물을 저장할 폴더 생성
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    # 테스트를 위한 더미 이미지 생성 (실제 사용 시 이 부분은 주석 처리하거나 삭제)
    if not os.path.exists(INPUT_IMAGE_FOLDER) or not os.listdir(INPUT_IMAGE_FOLDER):
        print(f"'{INPUT_IMAGE_FOLDER}' 폴더가 비어있어 테스트용 더미 이미지를 생성합니다.")
        os.makedirs(INPUT_IMAGE_FOLDER, exist_ok=True)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(30):
            x = int(100 + 200 * np.sin(i / 10.0))
            y = int(240 + 100 * np.cos(i / 10.0))
            cv2.putText(dummy_img, f'Dummy Frame {i+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(INPUT_IMAGE_FOLDER, f'frame_{i:03d}.png'), dummy_img)
        print("더미 이미지 생성이 완료되었습니다. 실제 이미지를 사용하려면 폴더 경로를 수정하세요.")


    # 메인 함수 실행
    track_hand_path_from_folder(
        input_folder=INPUT_IMAGE_FOLDER,
        output_video_path=OUTPUT_VIDEO,
        fps=10 # 초당 프레임 수 조절 가능
    )