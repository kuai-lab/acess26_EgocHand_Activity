
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # import matplotlib
# # matplotlib.use('Agg')  # GUI 없이 저장만

# # HAND_CONNECTIONS = [
# #     (0, 1), (1, 2), (2, 3), (3, 4),
# #     (0, 5), (5, 6), (6, 7), (7, 8),
# #     (0, 9), (9, 10), (10, 11), (11, 12),
# #     (0, 13), (13, 14), (14, 15), (15, 16),
# #     (0, 17), (17, 18), (18, 19), (19, 20)
# # ]

# # # CSV 경로
# # csv_path = "/home/jyseo/hand_journal/data_MHAV/assemble/hg/assemble_hexNut-bigBolt3_hand_hg/pose_2d_mp_csv/processed_270_480/keypoint_50.csv"
# # df = pd.read_csv(csv_path)
# # keypoints = df[["x", "y", "z"]].values

# # # 필요한 경우 x 좌표 좌우 반전 (예: 640 해상도 기준이면)
# # # keypoints[:, 0] = 640 - keypoints[:, 0]  # ← 왼손 좌우반전 필요 시

# # x, y, z = keypoints[:, 0], keypoints[:, 1], -keypoints[:, 2]

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(x, y, z, c='r', s=40)

# # # 오른손 기준 연결
# # for i, j in HAND_CONNECTIONS:
# #     ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue')

# # # 왼손 기준 연결 (21번 오프셋)
# # for i, j in HAND_CONNECTIONS:
# #     i_l, j_l = i + 21, j + 21
# #     ax.plot([x[i_l], x[j_l]], [y[i_l], y[j_l]], [z[i_l], z[j_l]], color='green')


# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.invert_yaxis()

# # # 📌 여기 추가: 깊이 기준 측면 시점 (정면에서 보는 느낌)
# # ax.view_init(elev=10, azim=270)  # elev: 위에서 본 각도, azim: 좌우 회전 각도

# # plt.title("3D Hand Skeleton (Left or Right)")
# # plt.tight_layout()
# # plt.savefig("hand_3d_skeleton.png")
# # print("✅ Saved: hand_3d_skeleton.png")


# ####--------------------------------------------------- 06/07
# # 몇가지 시퀀스 비디오폴더들에 대해서 테스트한 확장버전
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('Agg')
# from glob import glob

# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (0, 9), (9, 10), (10, 11), (11, 12),
#     (0, 13), (13, 14), (14, 15), (15, 16),
#     (0, 17), (17, 18), (18, 19), (19, 20)
# ]

# # 🔧 경로 설정
# csv_folder = "/home/jyseo/hand_journal/data_MHAV/carving"
# output_root = "hand_3d_outputs"

# csv_files = sorted(glob(os.path.join(csv_folder, "**", "processed_270_480", "keypoint_*.csv"), recursive=True))

# for csv_path in csv_files:
#     df = pd.read_csv(csv_path)
#     keypoints = df[["x", "y", "z"]].values
#     if keypoints.shape[0] != 42:
#         print(f"❌ Skipped (not 42 keypoints): {csv_path}")
#         continue

#     x, y, z = keypoints[:, 0], keypoints[:, 1], -keypoints[:, 2]  # z축 반전

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z, c='r', s=40)

#     # 오른손 연결
#     for i, j in HAND_CONNECTIONS:
#         ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue')

#     # 왼손 연결
#     for i, j in HAND_CONNECTIONS:
#         i_l, j_l = i + 21, j + 21
#         ax.plot([x[i_l], x[j_l]], [y[i_l], y[j_l]], [z[i_l], z[j_l]], color='green')

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.invert_yaxis()
#     ax.view_init(elev=10, azim=270)

#     # 🔁 상대 경로 추출 (assemble 이후)
#     rel_path = os.path.relpath(csv_path, csv_folder)  # 예: hg/assemble_hexNut.../keypoint_50.csv
#     rel_dir = os.path.dirname(rel_path)               # 디렉토리만
#     out_dir = os.path.join(output_root, rel_dir)
#     os.makedirs(out_dir, exist_ok=True)

#     fname = os.path.basename(csv_path).replace(".csv", ".png")
#     save_path = os.path.join(out_dir, fname)

#     plt.title(fname)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ Saved: {save_path}")






# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('Agg')  # GUI 없이 저장만

# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (0, 9), (9, 10), (10, 11), (11, 12),
#     (0, 13), (13, 14), (14, 15), (15, 16),
#     (0, 17), (17, 18), (18, 19), (19, 20)
# ]

# # CSV 경로
# csv_path = "/home/jyseo/hand_journal/data_MHAV/assemble/hg/assemble_hexNut-bigBolt3_hand_hg/pose_2d_mp_csv/processed_270_480/keypoint_50.csv"
# df = pd.read_csv(csv_path)
# keypoints = df[["x", "y", "z"]].values

# # 필요한 경우 x 좌표 좌우 반전 (예: 640 해상도 기준이면)
# # keypoints[:, 0] = 640 - keypoints[:, 0]  # ← 왼손 좌우반전 필요 시

# x, y, z = keypoints[:, 0], keypoints[:, 1], -keypoints[:, 2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='r', s=40)

# # 오른손 기준 연결
# for i, j in HAND_CONNECTIONS:
#     ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue')

# # 왼손 기준 연결 (21번 오프셋)
# for i, j in HAND_CONNECTIONS:
#     i_l, j_l = i + 21, j + 21
#     ax.plot([x[i_l], x[j_l]], [y[i_l], y[j_l]], [z[i_l], z[j_l]], color='green')


# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.invert_yaxis()

# # 📌 여기 추가: 깊이 기준 측면 시점 (정면에서 보는 느낌)
# ax.view_init(elev=10, azim=270)  # elev: 위에서 본 각도, azim: 좌우 회전 각도

# plt.title("3D Hand Skeleton (Left or Right)")
# plt.tight_layout()
# plt.savefig("hand_3d_skeleton.png")
# print("✅ Saved: hand_3d_skeleton.png")


####--------------------------------------------------- 06/07
# 몇가지 시퀀스 비디오폴더들에 대해서 테스트한 확장버전
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
from glob import glob

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# 🔧 경로 설정
csv_folder = "/data/jyseo/data_MHAV/file/hg/file_woodBlock5_roundFile_hg"
output_root = "hand_3d_outputs_file"

csv_files = sorted(glob(os.path.join(csv_folder, "pose_2d_mp_csv", "**", "keypoint_*.csv"), recursive=True))

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    keypoints = df[["x", "y", "z"]].values
    if keypoints.shape[0] != 42:
        print(f"❌ Skipped (not 42 keypoints): {csv_path}")
        continue

    x, y, z = keypoints[:, 0], keypoints[:, 1], -keypoints[:, 2]  # z축 반전

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', s=40)

    # 오른손 연결
    for i, j in HAND_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='blue')

    # 왼손 연결
    for i, j in HAND_CONNECTIONS:
        i_l, j_l = i + 21, j + 21
        ax.plot([x[i_l], x[j_l]], [y[i_l], y[j_l]], [z[i_l], z[j_l]], color='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_yaxis()
    ax.view_init(elev=10, azim=270)

    # 🔁 상대 경로 추출 (assemble 이후)
    rel_path = os.path.relpath(csv_path, csv_folder)  # 예: hg/assemble_hexNut.../keypoint_50.csv
    rel_dir = os.path.dirname(rel_path)               # 디렉토리만
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.basename(csv_path).replace(".csv", ".png")
    save_path = os.path.join(out_dir, fname)

    plt.title(fname)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")
