# # #### 이게 마지막에 보던 코드다!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ######

# # import os
# # import cv2
# # import torch
# # import numpy as np
# # import glob
# # from tqdm import tqdm
# # import re
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from scipy.spatial.transform import Rotation as ScipyRotation
# # from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # def natural_sort_key(s):
# #     """숫자 순서에 맞는 정렬을 위한 헬퍼 함수"""
# #     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# # def create_cylinder_vertices(radius=0.02, height=0.1, n_points=20):
# #     """가상 실린더의 꼭지점들을 생성하는 헬퍼 함수"""
# #     theta = np.linspace(0, 2 * np.pi, n_points)
# #     x = radius * np.cos(theta)
# #     y = radius * np.sin(theta)
# #     z_bottom = np.full_like(x, -height / 2)
# #     z_top = np.full_like(x, height / 2)
# #     bottom_cap = np.vstack([x, y, z_bottom]).T
# #     top_cap = np.vstack([x, y, z_top]).T
# #     return bottom_cap, top_cap


# # def visualize_rotation_and_top_cap_view(input_folder: str, 
# #                                         output_video_3d_path: str,
# #                                         output_video_top_cap_path: str,
# #                                         output_graph_path: str,
# #                                         track_hand: str = 'right', 
# #                                         fps: int = 10):
# #     # --- 1. 파이프라인 초기화 ---
# #     print("WiLoR 파이프라인 초기화...")
# #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# #     dtype = torch.float32
# #     pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
# #     mano_faces = pipe.wilor_model.mano.faces

# #     # --- 2. 데이터 경로 설정 및 프레임 선택 ---
# #     image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
# #     # 프레임 인덱스 128 이후만 사용 
# #     image_files = image_files[128:]
# #     if not image_files: 
        
# #         print(f"경고: '{input_folder}'에서 이미지를 찾을 수 없습니다.")
# #         return

# # # 일부 프레임만 추출
# #     # num_frames_to_use = 300
# #     # total_frames = len(image_files)
    
# #     # if total_frames > num_frames_to_use:
# #     #     center_index = total_frames // 2
# #     #     half_slice = num_frames_to_use // 2
# #     #     start_index = center_index - half_slice
# #     #     end_index = start_index + num_frames_to_use
# #     #     print(f"총 {total_frames} 프레임 중, 가운데 {num_frames_to_use}개 ({start_index}~{end_index-1}번)를 사용합니다.")
# #     #     image_files = image_files[start_index:end_index]
# #     # else:
# #     #     print(f"총 프레임({total_frames}개)이 요청 프레임({num_frames_to_use}개)보다 적어 모든 프레임을 사용합니다.")

# #     # --- 3. 3D 손 자세 데이터 추출 ---
# #     print("데이터 사전 분석 중 (선택된 영상 스캔)...")
# #     all_vertices, all_keypoints = [], []
# #     for img_path in tqdm(image_files, desc="데이터 사전 스캔"):
# #         frame = cv2.imread(img_path)
# #         if frame is None: continue
# #         outputs = pipe.predict(frame)
# #         if outputs:
# #             for out in outputs:
# #                 if (track_hand == 'right' and out['is_right'] == 1) or (track_hand == 'left' and out['is_right'] == 0):
# #                     all_vertices.append(out["wilor_preds"]["pred_vertices"][0])
# #                     all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
# #                     break
# #     if not all_keypoints: 
# #         print("경고: 유효한 손 키포인트를 감지하지 못했습니다.")
# #         return

# #     # --- 4. 렌더링을 위한 전역 변수 설정 ---
# #     all_keypoints_np = np.array(all_keypoints)
# #     all_keypoints_np[:, :, 0] *= -1

# #     # all_vertices도 보정 (필요한 경우)
# #     all_vertices_np = np.array(all_vertices)
# #     all_vertices_np[:, :, 0] *= -1
# #     global_min = np.min(all_keypoints_np.reshape(-1, 3), axis=0)
# #     global_max = np.max(all_keypoints_np.reshape(-1, 3), axis=0)
# #     max_range = np.max(global_max - global_min) * 1.5
# #     center_point = (global_max + global_min) / 2

# #     # --- 5. 영상 및 변수 초기화 ---
# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     fig_size_3d = (10, 8); dpi_3d = 120
# #     width_3d, height_3d = fig_size_3d[0] * dpi_3d, fig_size_3d[1] * dpi_3d
# #     vout_3d = cv2.VideoWriter(output_video_3d_path, fourcc, fps, (width_3d, height_3d))
# #     fig_3d = plt.figure(figsize=fig_size_3d, dpi=dpi_3d)

# #     fig_size_2d = (8, 8); dpi_2d = 120
# #     width_2d, height_2d = fig_size_2d[0] * dpi_2d, fig_size_2d[1] * dpi_2d
# #     vout_2d = cv2.VideoWriter(output_video_top_cap_path, fourcc, fps, (width_2d, height_2d))
# #     fig_2d, ax_2d = plt.subplots(figsize=fig_size_2d, dpi=dpi_2d)

# #     cyl_radius = 0.02
# #     base_cylinder_bottom, base_cylinder_top = create_cylinder_vertices(radius=cyl_radius)
# #     accumulated_rotation = ScipyRotation.identity()
# #     last_grasp_points = None
# #     grasp_indices = [4, 8, 12] # 엄지, 검지, 중지 끝점
# #     angle_history = []
# #     signed_accumulated_angle_deg = 0.0
# #     last_palm_axis = None

# #     print("3D 전체 뷰 및 Top Cap 회전 뷰 영상 동시 생성 중...")
    
# #     # --- 6. 메인 렌더링 루프 ---
# #  # for 루프 시작
# #     for i, (vertices_3d, keypoints_3d) in enumerate(tqdm(zip(all_vertices, all_keypoints), total=len(all_keypoints))):
        
# #         # ===================================================================
# #         # ✨ 새로운 단순 방식: "손목과 가까운 면의 방향"을 기준으로 각도 부호 결정
# #         # ===================================================================

# #         current_grasp_points = keypoints_3d[grasp_indices]
# #         cylinder_center = current_grasp_points.mean(axis=0)

# #         if last_grasp_points is not None:
# #             # 1. 시각화용 실린더의 회전은 안정성 필터를 적용하여 이전처럼 계산
# #             delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
# #             angle_change_deg = np.rad2deg(np.linalg.norm(delta_rotation.as_rotvec()))
# #             if angle_change_deg < 30.0:
# #                 accumulated_rotation = delta_rotation * accumulated_rotation

# #             # 2. '손목과 가까운 면'을 찾기 위한 거리 계산
# #             wrist_pos = keypoints_3d[0]
# #             # 실린더의 윗면/아랫면 중심의 현재 3D 좌표
# #             top_face_center_3d = accumulated_rotation.apply([0, 0, 0.05]) + cylinder_center
# #             bottom_face_center_3d = accumulated_rotation.apply([0, 0, -0.05]) + cylinder_center
# #             dist_to_top = np.linalg.norm(wrist_pos - top_face_center_3d)
# #             dist_to_bottom = np.linalg.norm(wrist_pos - bottom_face_center_3d)

# #             # 3. 더 가까운 면의 '법선 벡터'를 회전 방향 판단의 기준으로 선택
# #             if dist_to_bottom < dist_to_top:
# #                 # 아랫면이 가까움 -> 기준 축은 아랫면의 법선 벡터 (실린더의 로컬 -Z 방향)
# #                 reference_axis = accumulated_rotation.apply([0, 0, -1])
# #             else:
# #                 # 윗면이 가까움 -> 기준 축은 윗면의 법선 벡터 (실린더의 로컬 +Z 방향)
# #                 reference_axis = accumulated_rotation.apply([0, 0, 1])

# #             # 4. 선택된 기준 축(reference_axis)을 이용해 회전 방향(부호) 결정
# #             rotvec = delta_rotation.as_rotvec()
# #             angle_increment_rad = np.linalg.norm(rotvec)
            
# #             # 시계방향을 (+)로 하기 위해 -np.sign 사용
# #             sign = -np.sign(np.dot(rotvec, reference_axis))
            
# #             signed_delta_angle_deg = np.rad2deg(angle_increment_rad) * sign
# #         else: # 첫 프레임
# #             signed_delta_angle_deg = 0.0
        
# #         # --- 누적 및 변수 업데이트 ---
# #         signed_accumulated_angle_deg += signed_delta_angle_deg
# #         angle_history.append(signed_accumulated_angle_deg)
# #         last_grasp_points = current_grasp_points

# #         # ... 이하 3D / 2D 렌더링 코드는 모두 동일 ...
# #         # 2D 뷰 텍스트는 signed_accumulated_angle_deg 를 사용
# #         # ax_2d.text(..., f'Angle: {signed_accumulated_angle_deg:.1f}°', ...)

# #         # ... 이하 3D / 2D 렌더링 코드는 모두 동일 ...
# #         # 2D 뷰 텍스트는 signed_accumulated_angle_deg 를 사용
# #         # ax_2d.text(..., f'Angle: {signed_accumulated_angle_deg:.1f}°', ...)

# #         # ... 이하 3D / 2D 렌더링 코드는 모두 동일 ...
# #         # 단, 2D 뷰 텍스트와 최종 그래프는 'signed_accumulated_angle_deg'를 사용합니다.
# #         # 시계방향을 (+)로 하려면 최종 값에 -1을 곱해줍니다.
# #         # ax_2d.text(..., f'Angle: {-signed_accumulated_angle_deg:.1f}°', ...)
# #         # 최종 그래프 생성시: plt.plot(-np.array(angle_history_in_degrees))


# #         # --- 6-2. 3D 전체 뷰 렌더링 ---
# #         # ax_3d = fig_3d.add_subplot(111, projection='3d')
# #         # ax_3d.clear()
# #         # ax_3d.view_init(elev=30., azim=-60 + 90 * (i / len(all_keypoints)))
# #         # ax_3d.plot_trisurf(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], triangles=mano_faces, cmap=plt.cm.Greys, alpha=0.3)
# #         # ax_3d.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], c='cyan', s=10)
        
# #         # rotated_bottom = accumulated_rotation.apply(base_cylinder_bottom) + cylinder_center
# #         # rotated_top = accumulated_rotation.apply(base_cylinder_top) + cylinder_center
        
# #         # ax_3d.plot(rotated_bottom[:, 0], rotated_bottom[:, 1], rotated_bottom[:, 2], color='black')
# #         # top_face = Poly3DCollection([rotated_top], facecolors='blue', edgecolors='black')
# #         # ax_3d.add_collection3d(top_face)
        
# #         # for p1, p2 in zip(rotated_bottom, rotated_top):
# #         #      ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', alpha=0.5)
        
# #         # ref_line_p1 = rotated_top[0]; ref_line_p2 = rotated_top[10] # 참조선 길이 조절
# #         # ax_3d.plot([ref_line_p1[0], ref_line_p2[0]], [ref_line_p1[1], ref_line_p2[1]], [ref_line_p1[2], ref_line_p2[2]], color='red', linewidth=4)
        
# #         # ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
# #         # ax_3d.set_xlim(center_point[0]-max_range/2, center_point[0]+max_range/2)
# #         # ax_3d.set_ylim(center_point[1]-max_range/2, center_point[1]+max_range/2)
# #         # ax_3d.set_zlim(center_point[2]-max_range/2, center_point[2]+max_range/2)
# #         # ax_3d.set_title("3D Perspective View")

# #         # fig_3d.canvas.draw()
# #         # img_buf_3d = fig_3d.canvas.buffer_rgba()
# #         # frame_rgba_3d = np.frombuffer(img_buf_3d, dtype=np.uint8).reshape(height_3d, width_3d, 4)
# #         # vout_3d.write(cv2.cvtColor(frame_rgba_3d, cv2.COLOR_RGBA2BGR))

# #         # # --- 6-3. Top Cap 회전 뷰 렌더링 ---
# #         ax_2d.clear()
# #         ax_2d.fill(base_cylinder_top[:, 0], base_cylinder_top[:, 1], color='black', zorder=1)
# #         ref_point_base = np.array([cyl_radius, 0, 0])
# #         ref_point_rotated = accumulated_rotation.apply(ref_point_base)
# #         ax_2d.plot([0, ref_point_rotated[0]], [0, ref_point_rotated[1]], color='blue', linewidth=4, zorder=2)
# #         ax_2d.scatter(0, 0, color='red', s=50, zorder=3)
# #         ax_2d.text(0.95, 0.05, f'Angle: {signed_accumulated_angle_deg:.1f}°', 
# #                    transform=ax_2d.transAxes, ha='right', va='bottom', fontsize=12,
# #                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
# #         plot_radius = cyl_radius * 1.5
# #         ax_2d.set_xlim(-plot_radius, plot_radius); ax_2d.set_ylim(-plot_radius, plot_radius)
# #         ax_2d.set_aspect('equal', adjustable='box')
# #         ax_2d.set_title("Top Cap Rotation View (Normalized)")
# #         ax_2d.grid(True, linestyle=':')

# #         fig_2d.canvas.draw()
# #         img_buf_2d = fig_2d.canvas.buffer_rgba()
# #         frame_rgba_2d = np.frombuffer(img_buf_2d, dtype=np.uint8).reshape(height_2d, width_2d, 4)
# #         vout_2d.write(cv2.cvtColor(frame_rgba_2d, cv2.COLOR_RGBA2BGR))

# #     # --- 7. 자원 해제 및 그래프 생성 ---
# #     plt.close(fig_3d); plt.close(fig_2d)
# #     vout_3d.release(); vout_2d.release()

# #     print("회전 각도 변화 그래프 생성 중...")
# #     fig_graph, ax_graph = plt.subplots(figsize=(12, 6), dpi=100)
# #     ax_graph.plot(angle_history, marker='.', linestyle='-', label='Accumulated Angle')
# #     ax_graph.set_title("Accumulated Rotation Angle Over Time (Palm-based)")
# #     ax_graph.set_xlabel("Frame Index (Detected)")
# #     ax_graph.set_ylabel("Accumulated Angle (Degrees, CW is Positive)")
# #     ax_graph.grid(True); ax_graph.legend()
# #     fig_graph.savefig(output_graph_path)
# #     plt.close(fig_graph)
    
# #     print("-" * 50)
# #     print("성공! 모든 파일이 아래 경로에 저장되었습니다:")
# #     print(f"  - 3D 전체 뷰: {output_video_3d_path}")
# #     print(f"  - Top Cap 회전 뷰: {output_video_top_cap_path}")
# #     print(f"  - 회전 각도 그래프: {output_graph_path}")


# # if __name__ == '__main__':
# #     # ==========================================================
# #     #                  사용자 설정 영역
# #     # ==========================================================
# #     seq_list = [
# #         # 여기에 순회할 시퀀스들을 추가하세요
# #         "assemble_hexNut-metalPlate2-bolt-woodBlock2_hand_hg"
# #         # 필요하면 더 추가
# #     ]

# #     base_dir = "../data_MHAV/assemble"
# #     output_dir = "./wiLo/video/palm_2/assemble"
# #     os.makedirs(output_dir, exist_ok=True)
# #     # ==========================================================

# #     for seq in seq_list:
# #         # hand_xx 추출
# #         parts = seq.split('_')
# #         subj = parts[-1]  # hand_jl, hand_pc, hand_sp 등

# #         # 입력 폴더 생성
# #         input_folder = os.path.join(base_dir, subj, seq, "RGB_undistorted")

# #         # 출력 파일명 지정
# #         OUTPUT_VIDEO_3D = os.path.join(output_dir, f"{seq}_3d_view.mp4")
# #         OUTPUT_VIDEO_TOP_CAP = os.path.join(output_dir, f"{seq}_top_cap_rotation.mp4")
# #         OUTPUT_GRAPH_IMAGE = os.path.join(output_dir, f"{seq}_rotation_graph.png")

# #         # 함수 호출
# #         visualize_rotation_and_top_cap_view(
# #             input_folder=input_folder,
# #             output_video_3d_path=OUTPUT_VIDEO_3D,
# #             output_video_top_cap_path=OUTPUT_VIDEO_TOP_CAP,
# #             output_graph_path=OUTPUT_GRAPH_IMAGE,
# #             track_hand='right',
# #             fps=10
# #         )



# import os
# import cv2
# import torch
# import numpy as np
# import glob
# from tqdm import tqdm
# import re
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as ScipyRotation
# from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# def create_cylinder_vertices(radius=0.02, height=0.1, n_points=20):
#     theta = np.linspace(0, 2 * np.pi, n_points)
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     z_bottom = np.full_like(x, -height / 2)
#     z_top = np.full_like(x, height / 2)
#     bottom_cap = np.vstack([x, y, z_bottom]).T
#     top_cap = np.vstack([x, y, z_top]).T
#     return bottom_cap, top_cap

# def visualize_rotation(input_folder, output_video_path, output_graph_path, track_hand='right', fps=10):
#     print("WiLoR 파이프라인 초기화...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)

#     image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
#     image_files = image_files[128:]

#     if not image_files:
#         print(f"이미지 없음: {input_folder}")
#         return

#     all_keypoints = []
#     for img_path in tqdm(image_files, desc="3D 키포인트 추출"):
#         frame = cv2.imread(img_path)
#         if frame is None:
#             continue
#         outputs = pipe.predict(frame)
#         if outputs:
#             for out in outputs:
#                 if (track_hand == 'right' and out['is_right'] == 1) or \
#                    (track_hand == 'left' and out['is_right'] == 0):
#                     all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
#                     break

#     if not all_keypoints:
#         print("손 키포인트 없음.")
#         return

#     all_keypoints = np.array(all_keypoints)
#     all_keypoints[:, :, 0] *= -1  # 좌우 플립

#     accumulated_rotation = ScipyRotation.identity()
#     last_grasp_points = None
#     grasp_indices = [4, 8, 12]
#     cyl_radius = 0.02
#     base_bottom, base_top = create_cylinder_vertices(cyl_radius)
#     angle_history = []
#     signed_accum_angle = 0.0
#     reference_axis = None

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
#     vout = cv2.VideoWriter(output_video_path, fourcc, fps, (8 * 120, 8 * 120))

#     for i, keypoints_3d in enumerate(tqdm(all_keypoints, desc="회전 계산 및 영상 생성")):
#         current_grasp_points = keypoints_3d[grasp_indices]

#         if last_grasp_points is not None:
#             delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
#             angle_change_deg = np.rad2deg(np.linalg.norm(delta_rotation.as_rotvec()))
#             if angle_change_deg < 30.0:
#                 accumulated_rotation = delta_rotation * accumulated_rotation

#             rotvec = delta_rotation.as_rotvec()
#             angle_inc_rad = np.linalg.norm(rotvec)

#             # 첫 프레임 palm normal 기반 reference axis 고정
#             if reference_axis is None:
#                 # palm normal = 손바닥 바깥 방향
#                 thumb = current_grasp_points[0]
#                 index = current_grasp_points[1]
#                 middle = current_grasp_points[2]

#                 v1 = index - thumb
#                 v2 = middle - thumb
#                 palm_normal = np.cross(v1, v2)
#                 palm_normal /= np.linalg.norm(palm_normal)

#                 reference_axis = palm_normal
#                 print(f"참조축 고정 (palm normal): {reference_axis}")

#             dot = np.dot(rotvec, reference_axis)
#             sign = np.sign(dot)
#             signed_delta_angle_deg = np.rad2deg(angle_inc_rad) * sign

#             # 디버그 출력
#             print(f"Frame {i}: angle_inc={np.rad2deg(angle_inc_rad):.2f}, dot={dot:.4f}, sign={sign}, accum={signed_accum_angle + signed_delta_angle_deg:.2f}")
#         else:
#             signed_delta_angle_deg = 0.0

#         signed_accum_angle += signed_delta_angle_deg
#         angle_history.append(signed_accum_angle)
#         last_grasp_points = current_grasp_points

#         # 2D 시각화
#         ax.clear()
#         ax.fill(base_top[:, 0], base_top[:, 1], color='black', zorder=1)
#         ref_pt = np.array([cyl_radius, 0, 0])
#         ref_rotated = accumulated_rotation.apply(ref_pt)
#         ax.plot([0, ref_rotated[0]], [0, ref_rotated[1]], color='blue', linewidth=4)
#         ax.scatter(0, 0, color='red', s=50)
#         ax.text(0.95, 0.05, f'Angle: {signed_accum_angle:.1f}°', 
#                 transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
#                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
#         ax.set_xlim(-0.03, 0.03)
#         ax.set_ylim(-0.03, 0.03)
#         ax.set_aspect('equal')
#         ax.grid(True)

#         fig.canvas.draw()
#         img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(8 * 120, 8 * 120, 4)
#         vout.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

#     vout.release()
#     plt.close(fig)

#     plt.figure(figsize=(12, 6))
#     plt.plot(angle_history, marker='.', label='Accumulated Angle')
#     plt.title("Accumulated Rotation Angle Over Time (Palm outward)")
#     plt.xlabel("Frame")
#     plt.ylabel("Angle (deg)")
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(output_graph_path)
#     plt.close()

#     print(f"완료! 영상: {output_video_path}, 그래프: {output_graph_path}")

# if __name__ == '__main__':
#     seq_list = [
#         "assemble_hexNut-bigBolt3_hand_jl",
#         "assemble_hexNut-bolt-curvedMetalPlate_hand_jl",
#         "assemble_nut-spacer-bigBolt2_hand_jl",
#         "assemble_hexNut-bolt-curvedMetalPlate_hand_kl",
#         "assemble_hexNut-bolt_hand_kl",
#         "assemble_hexNut-uBolt-metalPlate4_hand_kl",
#         "assemble_metalBlock3-spacer-hexBolt_hand_kl",
#         "assemble_hexNut-bigBolt_hand_pc",
#         "assemble_hexNut-spacer-bigBolt2_hand_pc",
#         "assemble_bigBolt3-hexNut_hand_sp",
#         "assemble_hexNut-bigBolt_hand_sp",
#         "assemble_hexNut-bolt-curvedMetalPlate_hand_sp",
#         "assemble_hexNut-spacer-bigBolt2_hand_sp",
#         "assemble_metalPlate5-metalBlock2_hand_sp",
#         "assemble_hexNut-bigBolt3_hand_hg",
#         "assemble_hexNut-bolt-curvedMetalPlate_hand_hg",
#         "assemble_hexNut-bolt_hand_hg",
#         "assemble_hexNut-metalPlate2-bolt-woodBlock2_hand_hg",
#         "assemble_hexNut-metalPlate4-uBolt_hand_hg",
#         "assemble_hexNut-washer-blackHexBolt-metalPlate2_hand_hg",
#         "assemble_hexNut-washer-metalPlateBolt-metalPlate_hand_hg",
#         "assemble_hexNut-washer-metalPlateBolt-woodBlock2_hand_hg",
#         "assemble_nut-spacer-bigBolt2_hand_hg"
#     ]

#     base_dir = "../data_MHAV/assemble"
#     output_dir = "./wiLo/video/palm_debug/assemble"
#     os.makedirs(output_dir, exist_ok=True)

#     for seq in seq_list:
#         parts = seq.split('_')
#         subj = parts[-1]  # hand_jl, hand_kl, etc
#         input_folder = os.path.join(base_dir, subj, seq, "RGB_undistorted")

#         output_video_path = os.path.join(output_dir, f"{seq}_rotation.mp4")
#         output_graph_path = os.path.join(output_dir, f"{seq}_rotation_graph.png")

#         print(f"\n=== Processing {seq} ===")
#         print(f"Input: {input_folder}")
#         print(f"Output Video: {output_video_path}")
#         print(f"Output Graph: {output_graph_path}\n")

#         visualize_rotation(
#             input_folder=input_folder,
#             output_video_path=output_video_path,
#             output_graph_path=output_graph_path
#         )





import os
import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRotation
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_cylinder_vertices(radius=0.02, height=0.1, n_points=20):
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_bottom = np.full_like(x, -height / 2)
    z_top = np.full_like(x, height / 2)
    bottom_cap = np.vstack([x, y, z_bottom]).T
    top_cap = np.vstack([x, y, z_top]).T
    return bottom_cap, top_cap

def visualize_rotation(input_folder, output_video_path, output_graph_path, track_hand='right', fps=10):
    print("WiLoR 파이프라인 초기화...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)

    image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
    image_files = image_files[128:]
    if not image_files:
        print(f"⚠ 이미지 없음: {input_folder}")
        return

    all_keypoints = []
    for img_path in tqdm(image_files, desc="키포인트 추출"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        outputs = pipe.predict(frame)
        if outputs:
            for out in outputs:
                if (track_hand == 'right' and out['is_right'] == 1) or (track_hand == 'left' and out['is_right'] == 0):
                    all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
                    break

    if not all_keypoints:
        print("⚠ 유효한 손 키포인트 없음.")
        return

    all_keypoints = np.array(all_keypoints)
    all_keypoints[:, :, 0] *= -1

    accumulated_rotation = ScipyRotation.identity()
    last_grasp_points = None
    grasp_indices = [4, 8, 12]
    cyl_radius = 0.02
    base_bottom, base_top = create_cylinder_vertices(cyl_radius)
    angle_history = []
    signed_accum_angle = 0.0
    reference_axis = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    vout = cv2.VideoWriter(output_video_path, fourcc, fps, (8 * 120, 8 * 120))

    for i, keypoints_3d in enumerate(tqdm(all_keypoints, desc="렌더링")):
        current_grasp_points = keypoints_3d[grasp_indices]

        if last_grasp_points is not None:
            delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
            angle_change_deg = np.rad2deg(np.linalg.norm(delta_rotation.as_rotvec()))
            if angle_change_deg < 30.0:
                accumulated_rotation = delta_rotation * accumulated_rotation

            rotvec = delta_rotation.as_rotvec()
            angle_inc_rad = np.linalg.norm(rotvec)

            if reference_axis is None:
                v1 = current_grasp_points[1] - current_grasp_points[0]
                v2 = current_grasp_points[2] - current_grasp_points[0]
                palm_normal = np.cross(v1, v2)
                palm_normal /= (np.linalg.norm(palm_normal) + 1e-8)
                reference_axis = palm_normal
                if np.dot(reference_axis, np.array([0, 0, 1])) < 0:
                    reference_axis = -reference_axis


            dot = np.dot(rotvec, reference_axis)
            sign = np.sign(dot)
            signed_delta_angle_deg = np.rad2deg(angle_inc_rad) * sign

            print(f"Frame {i}: Δ={np.rad2deg(angle_inc_rad):.2f}°, dot={dot:.4f}, sign={sign}, accum={signed_accum_angle + signed_delta_angle_deg:.2f}")
        else:
            signed_delta_angle_deg = 0.0

        signed_accum_angle += signed_delta_angle_deg
        angle_history.append(signed_accum_angle)
        last_grasp_points = current_grasp_points

        ax.clear()
        ax.fill(base_top[:, 0], base_top[:, 1], color='black', zorder=1)
        ref_pt = np.array([cyl_radius, 0, 0])
        ref_rotated = accumulated_rotation.apply(ref_pt)
        ax.plot([0, ref_rotated[0]], [0, ref_rotated[1]], color='blue', linewidth=4)
        ax.scatter(0, 0, color='red', s=50)
        ax.text(0.95, 0.05, f'Angle: {signed_accum_angle:.1f}°',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        ax.set_xlim(-0.03, 0.03)
        ax.set_ylim(-0.03, 0.03)
        ax.set_aspect('equal')
        ax.grid(True)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(8 * 120, 8 * 120, 4)
        vout.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

    vout.release()
    plt.close(fig)

    plt.figure(figsize=(12, 6))
    plt.plot(angle_history, marker='.', label='Accumulated Angle')
    plt.title("Accumulated Rotation Angle Over Time (Palm normal aligned)")
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_graph_path)
    plt.close()

    print(f"🎬 영상: {output_video_path}")
    print(f"📈 그래프: {output_graph_path}")

if __name__ == '__main__':
    seq_list = [
        # 여기에 시퀀스를 추가하세요
        "assemble_hexNut-bigBolt3_hand_jl",
        "assemble_hexNut-bolt-curvedMetalPlate_hand_jl",
        "assemble_nut-spacer-bigBolt2_hand_jl",
        "assemble_hexNut-bolt-curvedMetalPlate_hand_kl",
        "assemble_hexNut-bolt_hand_kl",
        "assemble_hexNut-uBolt-metalPlate4_hand_kl",
        "assemble_metalBlock3-spacer-hexBolt_hand_kl",
        "assemble_hexNut-bigBolt_hand_pc",
        "assemble_hexNut-spacer-bigBolt2_hand_pc",
        "assemble_bigBolt3-hexNut_hand_sp",
        "assemble_hexNut-bigBolt_hand_sp",
        "assemble_hexNut-bolt-curvedMetalPlate_hand_sp",
        "assemble_hexNut-spacer-bigBolt2_hand_sp",
        "assemble_metalPlate5-metalBlock2_hand_sp",
        "assemble_hexNut-bigBolt3_hand_hg",
        "assemble_hexNut-bolt-curvedMetalPlate_hand_hg",
        "assemble_hexNut-bolt_hand_hg",
        "assemble_hexNut-metalPlate2-bolt-woodBlock2_hand_hg",
        "assemble_hexNut-metalPlate4-uBolt_hand_hg",
        "assemble_hexNut-washer-blackHexBolt-metalPlate2_hand_hg",
        "assemble_hexNut-washer-metalPlateBolt-metalPlate_hand_hg",
        "assemble_hexNut-washer-metalPlateBolt-woodBlock2_hand_hg",
        "assemble_nut-spacer-bigBolt2_hand_hg"    
        ]
        
    base_dir = "../data_MHAV/assemble"
    output_dir = "./wiLo/video/final_palm_aligned"
    os.makedirs(output_dir, exist_ok=True)

    for seq in seq_list:
        parts = seq.split('_')
        subj = parts[-1]
        input_folder = os.path.join(base_dir, subj, seq, "RGB_undistorted")
        output_video_path = os.path.join(output_dir, f"{seq}_rotation.mp4")
        output_graph_path = os.path.join(output_dir, f"{seq}_rotation_graph.png")
        visualize_rotation(input_folder, output_video_path, output_graph_path)

