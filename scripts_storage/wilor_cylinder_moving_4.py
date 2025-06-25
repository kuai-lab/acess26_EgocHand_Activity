# import os
# import cv2
# import torch
# import numpy as np
# import glob
# from tqdm import tqdm
# import re
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
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

# def visualize_rotation_and_top_cap_view(input_folder: str, 
#                                         output_video_3d_path: str,
#                                         output_video_top_cap_path: str, 
#                                         track_hand: str = 'right', 
#                                         fps: int = 10,
#                                         enable_3d: bool = True):
#     print("WiLoR 파이프라인 초기화...")
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     dtype = torch.float32
#     pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
#     mano_faces = pipe.wilor_model.mano.faces

#     image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
#     if not image_files:
#         print("⚠ No image files found.")
#         return

#     all_vertices, all_keypoints = [], []
#     for img_path in tqdm(image_files, desc="데이터 스캔"):
#         frame = cv2.imread(img_path)
#         if frame is None: continue
#         outputs = pipe.predict(frame)
#         if outputs:
#             for out in outputs:
#                 if (track_hand == 'right' and out['is_right'] == 1) or (track_hand == 'left' and out['is_right'] == 0):
#                     all_vertices.append(out["wilor_preds"]["pred_vertices"][0])
#                     all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
#                     break
#     if not all_keypoints:
#         print("⚠ No valid keypoints found.")
#         return

#     all_keypoints_np = np.array(all_keypoints)
#     global_min = np.min(all_keypoints_np.reshape(-1, 3), axis=0)
#     global_max = np.max(all_keypoints_np.reshape(-1, 3), axis=0)
#     max_range = np.max(global_max - global_min) * 1.5
#     center_point = (global_max + global_min) / 2

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
#     # 3D 초기화 (조건부)
#     fig_size_3d = (10, 8); dpi_3d = 120
#     width_3d, height_3d = fig_size_3d[0] * dpi_3d, fig_size_3d[1] * dpi_3d
#     if enable_3d:
#         vout_3d = cv2.VideoWriter(output_video_3d_path, fourcc, fps, (width_3d, height_3d))
#         fig_3d = plt.figure(figsize=fig_size_3d, dpi=dpi_3d)
    
#     # 2D 초기화
#     fig_size_2d = (8, 8); dpi_2d = 120
#     width_2d, height_2d = fig_size_2d[0] * dpi_2d, fig_size_2d[1] * dpi_2d
#     vout_2d = cv2.VideoWriter(output_video_top_cap_path, fourcc, fps, (width_2d, height_2d))
#     fig_2d, ax_2d = plt.subplots(figsize=fig_size_2d, dpi=dpi_2d)

#     cyl_radius = 0.02
#     base_cylinder_bottom, base_cylinder_top = create_cylinder_vertices(radius=cyl_radius)
#     accumulated_rotation = ScipyRotation.identity()
#     last_grasp_points = None
#     grasp_indices = [4, 8, 12]
#     angle_values = []

#     print("렌더링 중...")
#     for i, (vertices_3d, keypoints_3d) in enumerate(tqdm(zip(all_vertices, all_keypoints), total=len(all_keypoints))):
#         current_grasp_points = keypoints_3d[grasp_indices]
#         if last_grasp_points is not None:
#             delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
#             accumulated_rotation = delta_rotation * accumulated_rotation
#         cylinder_center = current_grasp_points.mean(axis=0)
#         last_grasp_points = current_grasp_points

#         total_rotation_angle_rad = np.linalg.norm(accumulated_rotation.as_rotvec())
#         total_rotation_angle_deg = np.rad2deg(total_rotation_angle_rad)
#         angle_values.append(total_rotation_angle_deg)

#         if enable_3d:
#             ax_3d = fig_3d.add_subplot(111, projection='3d')
#             ax_3d.clear()
#             ax_3d.view_init(elev=30., azim=-60 + 90 * (i / len(all_keypoints)))
#             ax_3d.plot_trisurf(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], triangles=mano_faces, cmap=plt.cm.Greys, alpha=0.3)
#             ax_3d.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], c='cyan', s=10)
#             rotated_bottom = accumulated_rotation.apply(base_cylinder_bottom) + cylinder_center
#             rotated_top = accumulated_rotation.apply(base_cylinder_top) + cylinder_center
#             ax_3d.plot(rotated_bottom[:, 0], rotated_bottom[:, 1], rotated_bottom[:, 2], color='black')
#             ax_3d.plot(rotated_top[:, 0], rotated_top[:, 1], rotated_top[:, 2], color='black')
#             for p1, p2 in zip(rotated_bottom, rotated_top):
#                 ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', alpha=0.5)
#             ref_line_p1 = rotated_bottom[0]; ref_line_p2 = rotated_top[0]
#             ax_3d.plot([ref_line_p1[0], ref_line_p2[0]], [ref_line_p1[1], ref_line_p2[1]], [ref_line_p1[2], ref_line_p2[2]], color='blue', linewidth=4)
#             ax_3d.set_xlim(center_point[0] - max_range/2, center_point[0] + max_range/2)
#             ax_3d.set_ylim(center_point[1] - max_range/2, center_point[1] + max_range/2)
#             ax_3d.set_zlim(center_point[2] - max_range/2, center_point[2] + max_range/2)

#             fig_3d.canvas.draw()
#             img_buf_3d = fig_3d.canvas.buffer_rgba()
#             frame_rgba_3d = np.frombuffer(img_buf_3d, dtype=np.uint8).reshape(height_3d, width_3d, 4)
#             frame_bgr_3d = cv2.cvtColor(frame_rgba_3d, cv2.COLOR_RGBA2BGR)
#             vout_3d.write(frame_bgr_3d)

#         ax_2d.clear()
#         ax_2d.plot(base_cylinder_top[:, 0], base_cylinder_top[:, 1], color='black')
#         ref_point_base = np.array([cyl_radius, 0, 0])
#         ref_point_rotated = accumulated_rotation.apply(ref_point_base)
#         ax_2d.plot([0, ref_point_rotated[0]], [0, ref_point_rotated[1]], color='blue', linewidth=4)
#         ax_2d.scatter(0, 0, color='red', s=50)
#         ax_2d.text(0.95, 0.95, f'Angle: {total_rotation_angle_deg:.1f}°',
#                    transform=ax_2d.transAxes, ha='right', va='top',
#                    bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))
#         ax_2d.set_xlim(-cyl_radius * 1.5, cyl_radius * 1.5)
#         ax_2d.set_ylim(-cyl_radius * 1.5, cyl_radius * 1.5)
#         ax_2d.set_aspect('equal')
#         ax_2d.grid(True, linestyle=':')

#         fig_2d.canvas.draw()
#         img_buf_2d = fig_2d.canvas.buffer_rgba()
#         frame_rgba_2d = np.frombuffer(img_buf_2d, dtype=np.uint8).reshape(height_2d, width_2d, 4)
#         frame_bgr_2d = cv2.cvtColor(frame_rgba_2d, cv2.COLOR_RGBA2BGR)
#         vout_2d.write(frame_bgr_2d)

#     if enable_3d:
#         plt.close(fig_3d)
#         vout_3d.release()
#     plt.close(fig_2d)
#     vout_2d.release()

#     print("✅ 영상 저장 완료.")
#     plt.figure(figsize=(10, 6))
#     plt.plot(angle_values, marker='o', linestyle='-', color='blue')
#     plt.xlabel("Frame Index")
#     plt.ylabel("Accumulated Rotation Angle (degrees)")
#     plt.title("Frame-wise Accumulated Rotation Angle")
#     plt.grid(True)
#     plot_path = output_video_top_cap_path.replace('.mp4', '_rotation_angle_plot.png')
#     plt.savefig(plot_path, dpi=300)
#     plt.close()
#     print(f"✅ Angle plot saved: {plot_path}")

# if __name__ == '__main__':
#     INPUT_IMAGE_FOLDER = "../data_MHAV/unassemble/jl/unassemble_hexNut-bigBolt_hand_jl/RGB_undistorted"
#     output_dir = "./wiLo/video/final_dual_view"
#     os.makedirs(output_dir, exist_ok=True)
#     video_name = os.path.basename(os.path.dirname(INPUT_IMAGE_FOLDER))
#     OUTPUT_VIDEO_3D = os.path.join(output_dir, f"{video_name}_3d_view.mp4")
#     OUTPUT_VIDEO_TOP_CAP = os.path.join(output_dir, f"{video_name}_top_cap_rotation.mp4")

#     visualize_rotation_and_top_cap_view(
#         input_folder=INPUT_IMAGE_FOLDER,
#         output_video_3d_path=OUTPUT_VIDEO_3D,
#         output_video_top_cap_path=OUTPUT_VIDEO_TOP_CAP,
#         track_hand='right',
#         fps=10,
#         enable_3d=False  # 3D 렌더링 OFF로 설정
#     )


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

def visualize_rotation_and_top_cap_view(input_folder: str, 
                                        output_video_top_cap_path: str, 
                                        track_hand: str = 'right', 
                                        fps: int = 10):
    print("WiLoR 파이프라인 초기화...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

    image_files = sorted(glob.glob(os.path.join(input_folder, '*.jpg')), key=natural_sort_key)
    if not image_files:
        print("⚠ No image files found.")
        return

    all_keypoints = []
    for img_path in tqdm(image_files, desc="데이터 스캔"):
        frame = cv2.imread(img_path)
        if frame is None: continue
        outputs = pipe.predict(frame)
        if outputs:
            for out in outputs:
                if (track_hand == 'right' and out['is_right'] == 1) or (track_hand == 'left' and out['is_right'] == 0):
                    all_keypoints.append(out["wilor_preds"]["pred_keypoints_3d"][0])
                    break
    if not all_keypoints:
        print("⚠ No valid keypoints found.")
        return

    fig_size_2d = (8, 8); dpi_2d = 120
    width_2d, height_2d = fig_size_2d[0] * dpi_2d, fig_size_2d[1] * dpi_2d
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout_2d = cv2.VideoWriter(output_video_top_cap_path, fourcc, fps, (width_2d, height_2d))
    fig_2d, ax_2d = plt.subplots(figsize=fig_size_2d, dpi=dpi_2d)

    cyl_radius = 0.02
    base_cylinder_top = create_cylinder_vertices(radius=cyl_radius)[1]
    accumulated_rotation = ScipyRotation.identity()
    last_grasp_points = None
    grasp_indices = [4, 8, 12]
    angle_values = []
    cumulative_angle_deg = 0

    # 기준 회전축 (예: Z축) → 필요시 변경
    reference_axis = np.array([0, 0, 1])

    for i, keypoints_3d in enumerate(tqdm(all_keypoints, desc="렌더링 중")):
        current_grasp_points = keypoints_3d[grasp_indices]
        if last_grasp_points is not None:
            delta_rotation, _ = ScipyRotation.align_vectors(current_grasp_points, last_grasp_points)
            rotvec = delta_rotation.as_rotvec()
            angle_increment = np.rad2deg(np.linalg.norm(rotvec))

            # 방향 부호 결정: rotvec의 방향과 기준축 dot product
            sign = np.sign(np.dot(rotvec, reference_axis))
            angle_increment_signed = angle_increment * sign

            cumulative_angle_deg += angle_increment_signed
            accumulated_rotation = delta_rotation * accumulated_rotation
        else:
            angle_increment_signed = 0

        last_grasp_points = current_grasp_points
        angle_values.append(cumulative_angle_deg)

        # === Top Cap 뷰 ===
        ax_2d.clear()
        ax_2d.plot(base_cylinder_top[:, 0], base_cylinder_top[:, 1], color='black')
        ref_point_base = np.array([cyl_radius, 0, 0])
        ref_point_rotated = accumulated_rotation.apply(ref_point_base)
        ax_2d.plot([0, ref_point_rotated[0]], [0, ref_point_rotated[1]], color='blue', linewidth=4)
        ax_2d.scatter(0, 0, color='red', s=50)
        ax_2d.text(0.95, 0.95, f'Angle: {cumulative_angle_deg:.1f}°',
                   transform=ax_2d.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))
        ax_2d.set_xlim(-cyl_radius * 1.5, cyl_radius * 1.5)
        ax_2d.set_ylim(-cyl_radius * 1.5, cyl_radius * 1.5)
        ax_2d.set_aspect('equal')
        ax_2d.grid(True, linestyle=':')

        fig_2d.canvas.draw()
        img_buf_2d = fig_2d.canvas.buffer_rgba()
        frame_rgba_2d = np.frombuffer(img_buf_2d, dtype=np.uint8).reshape(height_2d, width_2d, 4)
        frame_bgr_2d = cv2.cvtColor(frame_rgba_2d, cv2.COLOR_RGBA2BGR)
        vout_2d.write(frame_bgr_2d)

    plt.close(fig_2d)
    vout_2d.release()

    print("✅ 영상 저장 완료.")
    plt.figure(figsize=(10, 6))
    plt.plot(angle_values, marker='o', linestyle='-', color='blue', markersize=3)
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Rotation Angle (degrees)")
    plt.title("Frame-wise Cumulative Rotation Angle (Direction-Aware)")
    plt.grid(True)
    plot_path = output_video_top_cap_path.replace('.mp4', '_rotation_angle_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Angle plot saved: {plot_path}")
if __name__ == '__main__':
    output_dir = "./wiLo/video/final_dual_view_unscrew"
    os.makedirs(output_dir, exist_ok=True)

    seq_list = [
        "unscrew_woodChunk-roundHeadScrew_screwdriver_jl",
        "unscrew_woodChunk-countersunk_screwdriver_pc",
        "unscrew_woodBlock2-counterchunk_screwdriver-largeBlackClamp_sp",
        "unscrew_woodChunk-roundHeadScrew_screwdriver_sp",
        "unscrew_woodBlock-counterchunk_screwdriver_hg",
        "unscrew_woodBlock-counterchunk_screwdriver-largeBlackClamp_hg"
    ]

    for seq in seq_list:
        parts = seq.split('_')
        subj = parts[-1]  # 예: unscrew_woodChunk-roundHeadScrew_screwdriver_jl → jl
        INPUT_IMAGE_FOLDER = f"../data_MHAV/unscrew/{subj}/{seq}/RGB_undistorted"
        OUTPUT_VIDEO_TOP_CAP = os.path.join(output_dir, f"{seq}_top_cap_rotation.mp4")

        visualize_rotation_and_top_cap_view(
            input_folder=INPUT_IMAGE_FOLDER,
            output_video_top_cap_path=OUTPUT_VIDEO_TOP_CAP,
            track_hand='right',
            fps=10
        )