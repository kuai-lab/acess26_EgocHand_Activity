import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as ScipyRotation
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# --- 유틸리티 및 헬퍼 함수 ---

def natural_sort_key(s):
    """숫자 순서에 맞는 정렬을 위한 헬퍼 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def save_rotation_graph(rotation_data, output_path, title="Rotation Angle per Frame",
                        xlabel="Frame Index", ylabel="Rotation (degrees)", figsize=(12, 6)):
    """
    회전 각도 기록 데이터를 받아 그래프를 그리고 이미지 파일로 저장합니다.
    """
    if not rotation_data:
        print(f"경고: 그래프를 그릴 데이터가 없습니다. '{output_path}' 파일이 생성되지 않습니다.")
        return
    try:
        plt.figure(figsize=figsize)
        plt.plot(rotation_data, marker='.', linestyle='-', markersize=4, label='Rotation Angle')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle=':')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ 회전 각도 그래프 저장 완료: {output_path}")
    except Exception as e:
        print(f"오류: 그래프 저장에 실패했습니다. 원인: {e}")

def get_view_angles_from_normal(normal_vector):
    """법선 벡터로부터 matplotlib의 elev, azim 각도를 계산합니다."""
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    nx, ny, nz = normal_vector
    azim = np.rad2deg(np.arctan2(ny, nx))
    elev = np.rad2deg(np.arcsin(nz))
    return elev, azim

def create_transformed_cylinder(center, direction, length, radius, n_points=20, roll_angle_deg=0.0):
    """
    주어진 중심과 방향을 축으로 하고, 축에 대한 회전(roll)이 적용된 원기둥을 생성합니다.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    z = np.linspace(-length / 2, length / 2, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    start_vec = np.array([0.0, 0.0, 1.0])
    target_vec = direction / np.linalg.norm(direction)
    rot_align, _ = ScipyRotation.align_vectors([target_vec], [start_vec])
    rot_roll = ScipyRotation.from_euler('z', roll_angle_deg, degrees=True)
    final_rot = rot_align * rot_roll
    
    points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    transformed_points = final_rot.apply(points) + center
    
    X_transformed = transformed_points[:, 0].reshape(x_grid.shape)
    Y_transformed = transformed_points[:, 1].reshape(y_grid.shape)
    Z_transformed = transformed_points[:, 2].reshape(z_grid.shape)
    
    return X_transformed, Y_transformed, Z_transformed

def calculate_perpendicular_connector(point, axis_origin, axis_vector):
    """
    한 점에서 축(선)에 내린 수선에 해당하는 벡터와 '수선의 발' 좌표를 계산합니다.
    """
    origin_to_point_vector = point - axis_origin
    epsilon = 1e-6
    dot_product_axis = np.dot(axis_vector, axis_vector)
    if dot_product_axis < epsilon:
        return np.zeros(3), point
    projection_scalar = np.dot(origin_to_point_vector, axis_vector) / dot_product_axis
    foot_on_axis = axis_origin + (projection_scalar * axis_vector)
    perpendicular_vector = point - foot_on_axis
    return perpendicular_vector, foot_on_axis

# --- 데이터 처리 함수 ---

def load_and_select_image_files(input_folder, num_frames_to_use=50):
    """폴더에서 이미지를 로드하고 지정된 수의 프레임을 선택합니다."""
    print(f"이미지 파일 로드 중: {input_folder}")
    image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
    if not image_files:
        raise FileNotFoundError(f"경고: '{input_folder}'에서 이미지를 찾을 수 없습니다.")
    
    # num_frames_to_use가 전체 프레임보다 많으면 모든 프레임을 사용하도록 수정
    if len(image_files) < num_frames_to_use:
        print(f"총 프레임({len(image_files)}개)이 요청 프레임({num_frames_to_use}개)보다 적어 모든 프레임을 사용합니다.")
        return image_files
    else:
        print(f"총 {len(image_files)} 프레임 중, 처음부터 {num_frames_to_use}개의 프레임을 사용합니다.")
        return image_files[:num_frames_to_use]


def extract_3d_hand_data(pipe, image_files, track_hand='right'):
    """이미지 목록에서 3D 손 꼭짓점과 키포인트를 추출합니다."""
    print("데이터 사전 분석 중 (선택된 영상 스캔)...")
    all_vertices, all_keypoints = [], []
    for img_path in tqdm(image_files, desc="데이터 사전 스캔"):
        frame = cv2.imread(img_path)
        if frame is None: continue
        outputs = pipe.predict(frame)
        if outputs:
            for out in outputs:
                is_right_hand = out['is_right'] == 1
                if (track_hand == 'right' and is_right_hand) or (track_hand == 'left' and not is_right_hand):
                    all_vertices.append(out["wilor_preds"]["pred_vertices"][0])
                    all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
                    break
    if not all_keypoints:
        raise ValueError("경고: 유효한 손 키포인트를 감지하지 못했습니다.")
    return all_vertices, all_keypoints

# --- 3D 플로팅(그리기) 함수 ---

def plot_hand_mesh(ax, vertices, faces):
    """3D 축에 손 메쉬를 그립니다."""
    mesh = Poly3DCollection(vertices[faces], alpha=0.5, facecolor='cyan')
    mesh.set_edgecolor('k')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)

def plot_cylinder_with_guidelines(ax, center, direction, length, radius, n_points=20, roll_angle_deg=0.0):
    """방향을 나타내는 원기둥과 120도 간격의 기준선을 그립니다."""
    X_cyl, Y_cyl, Z_cyl = create_transformed_cylinder(center, direction, length, radius, n_points, roll_angle_deg=roll_angle_deg)
    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='gray', alpha=0.4, linewidth=0)
    
    line_angles_rad = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    line_colors = ['red', 'yellow', 'green']
    thetas = np.linspace(0, 2 * np.pi, n_points)

    for angle, color in zip(line_angles_rad, line_colors):
        idx = np.argmin(np.abs(thetas - angle))
        ax.plot(X_cyl[:, idx], Y_cyl[:, idx], Z_cyl[:, idx], color=color, linewidth=2.5)

def setup_3d_plot(ax, center, plot_range, title):
    """3D 플롯의 라벨, 제목, 축 범위, 시야각 등을 설정합니다."""
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    ax.set_xlim(center[0] - plot_range, center[0] + plot_range)
    ax.set_ylim(center[1] - plot_range, center[1] + plot_range)
    ax.set_zlim(center[2] - plot_range, center[2] + plot_range)
    ax.set_box_aspect([1,1,1])

# --- 메인 실행 함수 ---

def visualize_hand_to_video(input_folder: str, output_video_path: str, video_name: str,
                              track_hand: str = 'right', fps: int = 15, num_frames: int = 50):
    """
    일련의 이미지에서 손의 3D 자세를 추정하고,
    손의 방향을 나타내는 원기둥을 포함하여 동적 뷰 비디오로 렌더링합니다.
    """
    # 1. 파이프라인 초기화
    print("WiLoR 파이프라인 초기화...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)

    # 2. 데이터 로드 및 전처리
    image_files = load_and_select_image_files(input_folder, num_frames_to_use=num_frames)
    all_vertices, all_keypoints = extract_3d_hand_data(pipe, image_files, track_hand)
    
    # 3. 비디오 설정
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-70)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print("3D 시각화 및 동적 뷰 비디오 생성 시작...")
    plot_range = 0.15
    
    # --- 회전 계산을 위한 변수 초기화 ---
    last_grasp_points = None
    angle_history = []
    signed_accumulated_angle_deg = 0.0

    # 4. 각 프레임 렌더링 및 비디오 저장
    for i, (vertices, keypoints_3d) in enumerate(tqdm(zip(all_vertices, all_keypoints), total=len(all_keypoints), desc="비디오 프레임 렌더링")):
        ax.cla()

        # 손바닥 중심과 손가락 방향 벡터 계산
        palm_center = (keypoints_3d[0] + keypoints_3d[5] + keypoints_3d[17]) / 3
        finger_center = (keypoints_3d[4] + keypoints_3d[8] + keypoints_3d[12]) / 3
        palm_to_finger_vector = finger_center - palm_center
        
        # 엄지, 검지, 중지 끝점에서 손 중심축으로 내린 수선의 발 계산
        perp_thumb_vector, foot_thumb_point = calculate_perpendicular_connector(point=keypoints_3d[4], axis_origin=palm_center, axis_vector=palm_to_finger_vector)
        perp_index_vector, foot_index_point = calculate_perpendicular_connector(point=keypoints_3d[8], axis_origin=palm_center, axis_vector=palm_to_finger_vector)
        perp_middle_vector, foot_middle_point = calculate_perpendicular_connector(point=keypoints_3d[12], axis_origin=palm_center, axis_vector=palm_to_finger_vector)
        
        cylinder_length = np.linalg.norm(palm_to_finger_vector)
        cylinder_radius = 0.02
        thumb_vector_on_cylinder = (perp_thumb_vector / np.linalg.norm(perp_thumb_vector)) * cylinder_radius
        index_vector_on_cylinder = (perp_index_vector / np.linalg.norm(perp_index_vector)) * cylinder_radius
        middle_vector_on_cylinder = (perp_middle_vector / np.linalg.norm(perp_middle_vector)) * cylinder_radius

        point_thumb_on_cylinder_surface = foot_thumb_point + thumb_vector_on_cylinder
        point_index_on_cylinder_surface = foot_index_point + index_vector_on_cylinder
        point_middle_on_cylinder_surface = foot_middle_point + middle_vector_on_cylinder
        
        
        ax.scatter(
            point_thumb_on_cylinder_surface[0],   # X 좌표
            point_thumb_on_cylinder_surface[1],   # Y 좌표
            point_thumb_on_cylinder_surface[2],   # Z 좌표
            color='purple',          # 색상
            s=100,                # 점 크기
            edgecolor='white',    # 테두리 색상
            linewidth=1.5,        # 테두리 굵기
            depthshade=True,      # 깊이에 따라 음영 조절
            zorder=10             # 다른 요소들보다 위에 보이도록 설정
        )

        # 2. 검지손가락 방향의 점 찍기
        ax.scatter(
            point_index_on_cylinder_surface[0],
            point_index_on_cylinder_surface[1],
            point_index_on_cylinder_surface[2],
            color='purple',
            s=100,
            edgecolor='white',
            linewidth=1.5,
            depthshade=True,
            zorder=10
        )

        # 3. 중지손가락 방향의 점 찍기
        ax.scatter(
            point_middle_on_cylinder_surface[0],
            point_middle_on_cylinder_surface[1],
            point_middle_on_cylinder_surface[2],
            color='purple',
            s=100,
            edgecolor='white',
            linewidth=1.5,
            depthshade=True,
            zorder=10
        )
        
        
        # 회전 계산에 사용할 현재 프레임의 손가락 위치 점들을 정의
        # 이 점들은 손의 회전을 가장 잘 나타내는 점들입니다.
        # current_grasp_points = np.array([point_thumb_on_cylinder_surface, point_index_on_cylinder_surface, point_middle_on_cylinder_surface])
        current_grasp_points = np.array([keypoints_3d[4], keypoints_3d[8], keypoints_3d[12]])

        
        # 이전 프레임의 정보가 있을 경우에만 회전각을 계산
        if last_grasp_points is not None:
            # 두 점 집합 간의 최적 회전을 계산
            delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
            
            # 회전 벡터(rotvec)를 추출. 벡터의 방향이 회전축, 크기가 회전각(라디안)
            rotvec = delta_rotation.as_rotvec()
            
            # 회전축(rotvec)과 손의 중심축(palm_to_finger_vector)의 내적을 통해 회전 방향(부호) 결정
            # 두 벡터가 같은 방향이면 양수(반시계), 반대 방향이면 음수(시계)
            angle_increment_rad = np.linalg.norm(rotvec)
            sign = 0
            if angle_increment_rad > 1e-6: # 매우 작은 회전은 무시
                sign = np.sign(np.dot(rotvec, palm_to_finger_vector))

            # 부호를 적용한 프레임 간 회전 변화량 (단위: 도)
            signed_delta_angle_deg = np.rad2deg(angle_increment_rad) * sign
            
            # [수정 2] 계산된 변화량을 누적 각도에 더해줍니다.
            signed_accumulated_angle_deg += signed_delta_angle_deg
        
        # 값 기록 및 다음 프레임을 위한 변수 업데이트
        angle_history.append(signed_accumulated_angle_deg)
        last_grasp_points = current_grasp_points.copy() # copy()를 사용하여 안전하게 복사
        
        
        # --- 3D 시각화 ---
        cylinder_length = np.linalg.norm(palm_to_finger_vector)
        cylinder_radius = 0.02

        # 손 메쉬 그리기 (주석 처리됨)
        plot_hand_mesh(ax, vertices, pipe.wilor_model.mano.faces)
        
        # [수정 3] 원기둥 회전에 누적된 스칼라 각도(float)를 전달합니다.
        plot_cylinder_with_guidelines(ax, center=finger_center, direction=palm_to_finger_vector,
                                      length=cylinder_length, radius=cylinder_radius,
                                      roll_angle_deg=signed_accumulated_angle_deg)

        title = f'{video_name} (Frame {i})\nRotation: {signed_accumulated_angle_deg:.2f}°'
        setup_3d_plot(ax, palm_center, plot_range, title)
        
        # 프레임을 이미지로 변환하여 비디오에 쓰기
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_rgba = np.asarray(buf)
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        video_writer.write(img_bgr)

    # 5. 리소스 정리
    video_writer.release()
    plt.close(fig)
    print(f"✅ 동적 뷰 3D 비디오 저장 완료: {output_video_path}")
    
    # [수정 4] 비디오 생성 후, 누적된 회전 각도 데이터를 그래프로 저장합니다.
    output_graph_file = f"{os.path.dirname(output_video_path)}/{video_name}_rotation_graph.png"
    save_rotation_graph(angle_history, output_graph_file, title=f"'{video_name}' Hand Rotation Angle")
        
# --- 예제 실행 ---
if __name__ == '__main__':
    BASE_INPUT_DIR = "../data_MHAV/unscrew"
    OUTPUT_DIR = "./wiLo/0618/3d"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seq_list = [
        # 여기에 순회할 시퀀스들을 추가하세요
        "unscrew_woodChunk-roundHeadScrew_screwdriver_jl",
        "unscrew_woodChunk-countersunk_screwdriver_pc",
        "unscrew_woodBlock2-counterchunk_screwdriver-largeBlackClamp_sp",
        "unscrew_woodChunk-roundHeadScrew_screwdriver_sp",
        "unscrew_woodBlock-counterchunk_screwdriver_hg",
        "unscrew_woodBlock-counterchunk_screwdriver-largeBlackClamp_hg"
        # 필요하면 더 추가
    ]

    for seq_name in seq_list:
        subj = seq_name.split('_')[-1]
        input_folder = os.path.join(BASE_INPUT_DIR, subj, seq_name, "RGB_undistorted")
        output_video_file = os.path.join(OUTPUT_DIR, f"{seq_name}_3d.mp4")

        print(f"\n=== Processing {seq_name} ===")
        print(f"Input folder: {input_folder}")
        print(f"Output video: {output_video_file}")

        try:
            visualize_hand_to_video(
                input_folder=input_folder,
                output_video_path=output_video_file,
                video_name=seq_name,
                track_hand="right",
                fps=15,
                num_frames=99999  # 모든 프레임 사용
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️ 오류 발생 ({seq_name}): {e}")