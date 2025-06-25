import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm # 진행 상황을 보여주기 위한 라이브러리 (pip install tqdm)
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
import re
# --- 핵심 추가: 무지개색 경로를 위해 matplotlib.pyplot import ---
import matplotlib.pyplot as plt
import matplotlib

def natural_sort_key(s):
    """
    자연 정렬을 위한 키를 반환하는 함수.
    문자열에서 숫자 부분을 찾아 정수로 변환.
    예: "RGB_undistorted10.jpg" -> 10
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def visualize_3d_dual_output(input_folder: str, 
                               output_path_with_mesh: str, 
                               output_path_no_mesh: str, 
                               track_hand: str = 'right', 
                               fps: int = 30, 
                               tail_length: int = 50):
    """
    3D 손 시각화를 메시 포함/제외 두 가지 버전의 영상으로 동시에 생성합니다.
    """
    # ... (이전과 동일: 파이프라인 초기화, 데이터 사전 분석, 축 범위 설정) ...
    # (코드 중략)
    print("WiLoR 파이프라인 초기화...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    mano_faces = pipe.wilor_model.mano.faces

    image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
    if not image_files:
        print(f"오류: '{input_folder}'에서 이미지 파일을 찾을 수 없습니다.")
        return

    print("데이터 사전 분석 중...")
    all_vertices, all_keypoints = [], []
    for img_path in tqdm(image_files, desc="데이터 사전 스캔"):
        frame = cv2.imread(img_path)
        if frame is None: continue
        outputs = pipe.predict(frame)
        if outputs:
            for out in outputs:
                if (track_hand == 'right' and out['is_right'] == 1) or \
                   (track_hand == 'left' and out['is_right'] == 0):
                    all_vertices.append(out["wilor_preds"]["pred_vertices"][0])
                    all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
                    break
    
    if not all_vertices:
        print("경고: 추적할 손의 3D 데이터를 찾지 못했습니다.")
        return

    all_vertices_np = np.array(all_vertices)
    x_min, y_min, z_min = all_vertices_np.min(axis=(0, 1)); x_max, y_max, z_max = all_vertices_np.max(axis=(0, 1))
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 1.2
    x_mid, y_mid, z_mid = (x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2
    
    # --- 핵심 변경: 두 개의 VideoWriter 객체 생성 ---
    fig_size = (10, 8); dpi = 100
    width, height = fig_size[0] * dpi, fig_size[1] * dpi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    vout_with_mesh = cv2.VideoWriter(output_path_with_mesh, fourcc, fps, (width, height))
    vout_no_mesh = cv2.VideoWriter(output_path_no_mesh, fourcc, fps, (width, height))

    wrist_path_history = []
    num_frames = len(all_vertices)
    cmap = plt.get_cmap('jet')

    for i, (vertices_3d, keypoints_3d) in enumerate(tqdm(zip(all_vertices, all_keypoints), desc="3D 듀얼 영상 생성", total=num_frames)):
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30., azim=-60 + 90 * (i / num_frames))

        # 공통 요소 그리기 (경로, 벡터 등)
        wrist_3d = keypoints_3d[0]; thumb_tip_3d = keypoints_3d[4]; index_tip_3d = keypoints_3d[18]
        wrist_path_history.append(wrist_3d)
        
        start_index = max(0, len(wrist_path_history) - tail_length)
        path_tail = wrist_path_history[start_index:]
        if len(path_tail) > 1:
            for j in range(1, len(path_tail)):
                absolute_index = start_index + j
                p1, p2 = path_tail[j-1], path_tail[j]
                color = cmap(absolute_index / num_frames)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linestyle='--')

        thumb_vec = thumb_tip_3d - wrist_3d
        ax.quiver(wrist_3d[0], wrist_3d[1], wrist_3d[2], thumb_vec[0], thumb_vec[1], thumb_vec[2], color='b', arrow_length_ratio=0.3, label='Wrist-Thumb')
        index_vec = index_tip_3d - wrist_3d
        ax.quiver(wrist_3d[0], wrist_3d[1], wrist_3d[2], index_vec[0], index_vec[1], index_vec[2], color='m', arrow_length_ratio=0.3, label='Wrist-Index')
        ax.scatter(wrist_3d[0], wrist_3d[1], wrist_3d[2], color='r', s=50, label='Wrist', depthshade=False)
        ax.legend(loc='upper left')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m, Depth)')
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2); ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2); ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
        ax.set_title(f"3D Hand Pose (Frame {i+1})")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=num_frames); sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.1); cbar.set_label('Time (Frame #)')
        
        # --- 핵심 로직: 메시 포함/제외 프레임 각각 생성 ---
        
        # 1. 메시를 그려서 '메시 포함' 영상에 저장
        x, y, z = vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2]
        mesh_artist = ax.plot_trisurf(x, y, z, triangles=mano_faces, cmap=plt.cm.Greys, alpha=0.5)
        
        fig.canvas.draw()
        img_buf_with_mesh = fig.canvas.buffer_rgba()
        frame_rgba_with_mesh = np.frombuffer(img_buf_with_mesh, dtype=np.uint8).reshape(height, width, 4)
        frame_bgr_with_mesh = cv2.cvtColor(frame_rgba_with_mesh, cv2.COLOR_RGBA2BGR)
        vout_with_mesh.write(frame_bgr_with_mesh)
        
        # 2. 그렸던 메시를 제거하고 '메시 제외' 영상에 저장
        mesh_artist.remove()
        
        fig.canvas.draw()
        img_buf_no_mesh = fig.canvas.buffer_rgba()
        frame_rgba_no_mesh = np.frombuffer(img_buf_no_mesh, dtype=np.uint8).reshape(height, width, 4)
        frame_bgr_no_mesh = cv2.cvtColor(frame_rgba_no_mesh, cv2.COLOR_RGBA2BGR)
        vout_no_mesh.write(frame_bgr_no_mesh)
        
        plt.close(fig) # 메모리 해제

    # --- 핵심 변경: 두 개의 VideoWriter 객체 모두 해제 ---
    vout_with_mesh.release()
    vout_no_mesh.release()
    
    print("-" * 50)
    print("성공! 두 가지 버전의 3D 시각화 영상이 아래 경로에 저장되었습니다:")
    print(f"  - 메시 포함: {output_path_with_mesh}")
    print(f"  - 메시 제외: {output_path_no_mesh}")


if __name__ == '__main__':
    INPUT_IMAGE_FOLDER = "/data/jyseo/data_MHAV/assemble/hg/assemble_hexNut-bolt_hand_hg/RGB_undistorted"
    
    # --- 핵심 변경: 두 개의 출력 파일 경로 설정 ---
    output_dir = "/data/jyseo/functional_hand_type/direction_hist_results/gibson/trajectory_results/wilor/"
    OUTPUT_VIDEO_WITH_MESH = os.path.join(output_dir, "test_3d_with_mesh.mp4")
    OUTPUT_VIDEO_NO_MESH = os.path.join(output_dir, "test_3d_no_mesh.mp4")
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(INPUT_IMAGE_FOLDER) or not os.listdir(INPUT_IMAGE_FOLDER):
        print(f"경고: '{INPUT_IMAGE_FOLDER}' 폴더가 비어있거나 존재하지 않습니다.")
    else:
        visualize_3d_dual_output( # 함수 이름 변경
            input_folder=INPUT_IMAGE_FOLDER,
            output_path_with_mesh=OUTPUT_VIDEO_WITH_MESH,
            output_path_no_mesh=OUTPUT_VIDEO_NO_MESH,
            track_hand='right',
            fps=15
        )