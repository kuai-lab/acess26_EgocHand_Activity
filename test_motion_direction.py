# ## =============================================================================== ##
# ## ======= 06/15 Histogram of Hand Directions(HoHD) & Wrist rotations(HoWR) ====== ##

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch

# def normalize_tensor(tensor, dim=1):
#     norm = torch.norm(tensor, dim=dim, keepdim=True).clamp(min=1e-12)
#     return tensor / norm


# def load_keypoints_from_csv(csv_folder, max_frames=None):       # 양손 다 감지된 csv만 불러오게 
#     csv_paths = sorted(Path(csv_folder).glob("*.csv"))
#     if max_frames is not None:
#         csv_paths = csv_paths[:max_frames]
#     frames = []
#     for csv_path in csv_paths:
#         df = pd.read_csv(csv_path)
#         if df.shape[0] != 42:                               # 한 손만 존재하는 프레임 skip
#             continue
#         if df[['x', 'y', 'z']].abs().sum().sum() == 0:      # 42줄 모두 0으로 채워진 프레임 skip(손이 없는 경우)
#             continue
#         frame = df[['x', 'y', 'z']].values.astype(np.float32).reshape(2, 21, 3)
#         frames.append(frame)
#     if not frames:
#         raise ValueError(f"No valid CSV in {csv_folder}")
#     kp = np.stack(frames, axis=0)
#     kp = np.transpose(kp, (3, 0, 2, 1))  # (C, T, V, M)
#     kp = np.expand_dims(kp, axis=0)      # (1, C, T, V, M)
#     return torch.tensor(kp, dtype=torch.float32)


# def compute_direction_histogram(x, bins_theta=8, bins_phi=4):
#     B, C, T, V, M = x.shape
#     palm_idx = 0                                     # palm(손바닥인데 손목으로 지정되어있는거 참고) joint index
#     pos = x[:, :, :, palm_idx, :]  # (B, C, T, M)
    
#     # Compute direction vectors (offset L)
#     L = 10                                          # directional vector 추출하는 프레임 간격 offset 조정
#     dir_vec = pos[:, :, L:, :] - pos[:, :, :-L, :]  # directional 벡터 계산 
#     dir_vec = normalize_tensor(dir_vec, dim=1)      # (B, C, T', M), 단위 벡터로 정규화
    
#     # Convert to spherical coordinates
#     x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
#     theta = torch.atan2(y_, x_)                   # [-pi, pi] => (0, 2pi) = xy 평면의 방향각 
#     theta = (theta + np.pi) / (2 * np.pi)         # [0,1)
    
#     phi = torch.acos(torch.clamp(z_, -1.0, 1.0))  # [0, pi]  z축 기준 각도
#     phi = phi / np.pi  # [0,1)

#     # Bin => theta/phi를 히스토그램 bin 인덱스로 변환 (flat index)
#     theta_bins = (theta * bins_theta).long().clamp(0, bins_theta - 1)
#     phi_bins = (phi * bins_phi).long().clamp(0, bins_phi - 1)
#     bin_idx = theta_bins * bins_phi + phi_bins  # combined index

#     # Build histogram => 히스토그램 생성
#     hist = torch.zeros(B, M, bins_theta * bins_phi, device=x.device)
#     for b in range(B):
#         for m in range(M):
#             idx = bin_idx[b, :, m].view(-1)
#             hist[b, m].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    
#     # Normalize
#     hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
#     hist = hist.mean(dim=1)  # average over hands
#     return hist.cpu().numpy()

# # def compute_rotation_histogram(x, bins_theta=8, bins_phi=4):
# #     B, C, T, V, M = x.shape

# #     # wrist 위치
# #     wrist_pos = x[:, :, :, 0, :]  # (B, C, T, M)
    
# #     # palm 중심: MediaPipe 0(wrist) + 5,9,13,17 (metacarpal base) 평균
# #     palm_base_idx = [0, 5, 9, 13, 17]
# #     palm_pos = x[:, :, :, palm_base_idx, :].mean(dim=3)  # (B, C, T, M)

# #     # wrist -> palm 벡터
# #     dir_vec = palm_pos - wrist_pos  # (B, C, T, M)
# #     dir_vec = normalize_tensor(dir_vec, dim=1)

# #     # spherical coords
# #     x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
# #     theta = torch.atan2(y_, x_) 
# #     theta = (theta + np.pi) / (2 * np.pi)

# #     phi = torch.acos(torch.clamp(z_, -1.0, 1.0))
# #     phi = phi / np.pi

# #     # binning
# #     theta_bins = (theta * bins_theta).long().clamp(0, bins_theta - 1)
# #     phi_bins = (phi * bins_phi).long().clamp(0, bins_phi - 1)
# #     bin_idx = theta_bins * bins_phi + phi_bins

# #     # histogram
# #     hist = torch.zeros(B, M, bins_theta * bins_phi, device=x.device)
# #     for b in range(B):
# #         for m in range(M):
# #             idx = bin_idx[b, :, m].view(-1)
# #             hist[b, m].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

# #     hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
# #     hist = hist.mean(dim=1)
# #     return hist.cpu().numpy()


# #오른손만 
# def compute_rotation_histogram(x, bins_theta=8, bins_phi=4):
#     B, C, T, V, M = x.shape

#     wrist_pos = x[:, :, :, 0, 1:2]  # Hand 1만 선택 (M=1 slice)
    
#     palm_base_idx = [0, 5, 9, 13, 17]
#     palm_pos = x[:, :, :, palm_base_idx, 1:2].mean(dim=3)  # Hand 1만 선택

#     dir_vec = palm_pos - wrist_pos
#     dir_vec = normalize_tensor(dir_vec, dim=1)

#     x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
#     theta = torch.atan2(y_, x_)
#     theta = (theta + np.pi) / (2 * np.pi)

#     phi = torch.acos(torch.clamp(z_, -1.0, 1.0))
#     phi = phi / np.pi

#     theta_bins = (theta * bins_theta).long().clamp(0, bins_theta - 1)
#     phi_bins = (phi * bins_phi).long().clamp(0, bins_phi - 1)
#     bin_idx = theta_bins * bins_phi + phi_bins

#     hist = torch.zeros(B, 1, bins_theta * bins_phi, device=x.device)  # M=1
#     for b in range(B):
#         idx = bin_idx[b, :, 0].view(-1)  # M=1 slice
#         hist[b, 0].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

#     hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
#     return hist[:,0].cpu().numpy()  # M=1만 리턴


# def plot_histograms_by_class(tokens, labels, label_names, bins_theta=8, bins_phi=4):
#     classes = np.unique(labels)
#     save_dir = Path("./direction_hist_results")
#     save_dir.mkdir(exist_ok=True)

#     for c in classes:
#         cls_tokens = tokens[labels == c]
#         mean_hist = cls_tokens.mean(axis=0)
#         std_hist = cls_tokens.std(axis=0)

#         plt.figure(figsize=(10,6))
#         plt.bar(np.arange(len(mean_hist)), mean_hist, yerr=std_hist, alpha=0.7, label=f"{label_names[c]} mean ± std")
#         plt.title(f"Histogram of Hand Direction - {label_names[c]}")
#         plt.xlabel(f"Bin index (theta * {bins_phi} + phi)")
#         plt.ylabel("Normalized Frequency")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
        
#         filename = save_dir / f"{label_names[c]}_direction_histogram.png"
#         plt.savefig(filename, dpi=300)
#         print(f"Saved histogram plot: {filename}")
#         plt.close()


# def plot_histogram_single(hist, label_name, bins_theta=8, bins_phi=4):
#     mean_hist = hist.mean(axis=0)
#     std_hist = hist.std(axis=0)

#     save_dir = Path("./direction_hist_results")
#     save_dir.mkdir(exist_ok=True)

#     plt.figure(figsize=(10,6))
#     plt.bar(np.arange(len(mean_hist)), mean_hist, yerr=std_hist, alpha=0.7, label=f"{label_name} mean ± std")
#     plt.title(f"Histogram of Hand Direction - {label_name}")
#     plt.xlabel(f"Bin index (theta * {bins_phi} + phi)")
#     plt.ylabel("Normalized Frequency")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     filename = save_dir / f"{label_name}_direction_histogram.png"
#     plt.savefig(filename, dpi=300)
#     print(f"Saved histogram plot: {filename}")
#     plt.close()




# # # === 데이터 로드 ===
# # df = pd.read_excel("/data/jyseo/functional_hand_type/0612_screw_unscrew_filter.xlsx")
# # valid_rows = df[df.iloc[:, 1] == 'O']

# # sequence_names = valid_rows.iloc[:, 0].tolist()
# # base_dir = Path("../data_MHAV")
# # data_paths = {"screw": [], "unscrew": []}
# # for seq_name in sequence_names:
# #     if seq_name.startswith("screw"):
# #         label = "screw"
# #     elif seq_name.startswith("unscrew"):
# #         label = "unscrew"
# #     else:
# #         continue
# #     try:
# #         hand_type = seq_name.split("_")[-1]
# #         seq_path = base_dir / label / hand_type / seq_name / "pose_2d_mp_csv"
# #         if seq_path.exists():
# #             data_paths[label].append(seq_path)
# #     except:
# #         continue


# # === 여기서 시퀀스 경로를 직접 지정 ===
# # 예: seq_path = Path("../data_MHAV/screw/right/screw_01_right/pose_2d_mp_csv")
# seq_path = Path("../data_MHAV/unscrew/jl/unscrew_woodChunk-socketCapScrew_allenKey_jl/pose_2d_mp_csv")  # 이 부분을 원하는 시퀀스 경로로 수정
# # seq_path = Path("../data_MHAV/screw/jl/screw_woodChunk-socketCapScrew_allenKey_jl/pose_2d_mp_csv")
# # === feature 추출 및 플롯 ===
# with torch.no_grad():
#     try:
#         x = load_keypoints_from_csv(str(seq_path)).cuda()
#         hist = compute_rotation_histogram(x, bins_theta=8, bins_phi=4)
#         plot_histogram_single(hist, label_name=seq_path.parent.parent.name, bins_theta=8, bins_phi=4)
#     except Exception as e:
#         print(f"[ERROR] {seq_path}: {e}")



# # # # === feature 추출 ===
# # # tokens = []
# # # labels = []
# # # with torch.no_grad():
# # #     for cls, paths in data_paths.items():
# # #         for path in paths:
# # #             try:
# # #                 x = load_keypoints_from_csv(str(path)).cuda()
# # #                 hist = compute_direction_histogram(x, bins_theta=8, bins_phi=4)
# # #                 tokens.append(hist.squeeze(0))
# # #                 labels.append(0 if cls == "screw" else 1)
# # #             except Exception as e:
# # #                 print(f"[ERROR] {path}: {e}")
# # #                 continue

# # # tokens = np.stack(tokens)
# # # labels = np.array(labels)

# # # # === plot ===
# # # plot_histograms_by_class(tokens, labels, label_names={0: "screw", 1: "unscrew"}, bins_theta=8, bins_phi=4)



import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def normalize_tensor(tensor, dim=1):
    norm = torch.norm(tensor, dim=dim, keepdim=True).clamp(min=1e-12)
    return tensor / norm

def load_keypoints_from_csv(csv_folder, max_frames=None):
    csv_paths = sorted(Path(csv_folder).glob("*.csv"))
    if max_frames is not None:
        csv_paths = csv_paths[:max_frames]
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.shape[0] != 42:
            continue
        if df[['x', 'y', 'z']].abs().sum().sum() == 0:
            continue
        frame = df[['x', 'y', 'z']].values.astype(np.float32).reshape(2, 21, 3)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No valid CSV in {csv_folder}")
    kp = np.stack(frames, axis=0)
    kp = np.transpose(kp, (3, 0, 2, 1))  # (C, T, V, M)
    kp = np.expand_dims(kp, axis=0)      # (1, C, T, V, M)
    return torch.tensor(kp, dtype=torch.float32)


# histogram of wrist rotation
def compute_rotation_histogram(x, bins_theta=8, bins_phi=4):
    B, C, T, V, M = x.shape
    wrist_pos = x[:, :, :, 0, 1:2]
    palm_base_idx = [0, 5, 9, 13, 17]
    palm_pos = x[:, :, :, palm_base_idx, 1:2].mean(dim=3)
    dir_vec = palm_pos - wrist_pos
    dir_vec = normalize_tensor(dir_vec, dim=1)

    x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
    theta = torch.atan2(y_, x_)
    theta = (theta + np.pi) / (2 * np.pi)
    phi = torch.acos(torch.clamp(z_, -1.0, 1.0))
    phi = phi / np.pi

    theta_bins = (theta * bins_theta).long().clamp(0, bins_theta - 1)
    phi_bins = (phi * bins_phi).long().clamp(0, bins_phi - 1)
    bin_idx = theta_bins * bins_phi + phi_bins

    hist = torch.zeros(B, 1, bins_theta * bins_phi, device=x.device)
    for b in range(B):
        idx = bin_idx[b, :, 0].view(-1)
        hist[b, 0].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return hist[:,0].cpu().numpy()



# histogram of hand direction
def compute_direction_histogram(x, bins_theta=8, bins_phi=4, L=10):
    B, C, T, V, M = x.shape
    palm_idx = 0
    pos = x[:, :, :, palm_idx, 1:2]  # 오른손 palm (joint 0 = wrist)
    
    dir_vec = pos[:, :, L:, :] - pos[:, :, :-L, :]
    dir_vec = normalize_tensor(dir_vec, dim=1)

    x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
    theta = torch.atan2(y_, x_)
    theta = (theta + np.pi) / (2 * np.pi)
    phi = torch.acos(torch.clamp(z_, -1.0, 1.0))
    phi = phi / np.pi

    theta_bins = (theta * bins_theta).long().clamp(0, bins_theta - 1)
    phi_bins = (phi * bins_phi).long().clamp(0, bins_phi - 1)
    bin_idx = theta_bins * bins_phi + phi_bins

    hist = torch.zeros(B, 1, bins_theta * bins_phi, device=x.device)
    for b in range(B):
        idx = bin_idx[b, :, 0].view(-1)
        hist[b, 0].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    hist = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return hist[:,0].cpu().numpy()



def compute_theta_dynamics_rotation(x):
    B, C, T, V, M = x.shape
    wrist_pos = x[:, :, :, 0, 1:2]
    palm_base_idx = [0, 5, 9, 13, 17]
    palm_pos = x[:, :, :, palm_base_idx, 1:2].mean(dim=3)
    dir_vec = palm_pos - wrist_pos
    dir_vec = normalize_tensor(dir_vec, dim=1)

    x_, y_, z_ = dir_vec[:,0], dir_vec[:,1], dir_vec[:,2]
    theta = torch.atan2(y_, x_).cpu().numpy()

    dtheta = np.diff(theta, axis=1)
    cumulative_theta = dtheta.cumsum(axis=1)
    mean_dtheta = dtheta.mean()

    print(f"Mean dtheta: {mean_dtheta:.4f} -> {'반시계' if mean_dtheta > 0 else '시계' if mean_dtheta < 0 else '정체'}")

    save_dir = Path("./direction_hist_results")
    save_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(cumulative_theta[0,:,0])
    plt.title("Cumulative θ over time (right hand)")
    plt.xlabel("Frame")
    plt.ylabel("Cumulative θ (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "right_hand_cumulative_theta_screw.png", dpi=300)
    plt.close()

    return mean_dtheta, cumulative_theta


def plot_histogram_single(hist, label_name, bins_theta=8, bins_phi=4):
    mean_hist = hist.mean(axis=0)
    std_hist = hist.std(axis=0)

    save_dir = Path("./direction_hist_results")
    save_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(mean_hist)), mean_hist, yerr=std_hist, alpha=0.7, label=f"{label_name} mean ± std")
    plt.title(f"Rotation Histogram (right hand) - {label_name}")
    plt.xlabel(f"Bin index (theta * {bins_phi} + phi)")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = save_dir / f"{label_name}_rotation_histogram_screw.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved rotation histogram plot: {filename}")
    plt.close()

## ======= 실행 ======= ##
# seq_path = Path("../data_MHAV/unscrew/jl/unscrew_woodChunk-socketCapScrew_allenKey_jl/pose_2d_mp_csv")  # 원하는 경로 지정
seq_path = Path("../data_MHAV/screw/jl/screw_woodChunk-socketCapScrew_allenKey_jl/pose_2d_mp_csv")


label_name = seq_path.parent.parent.name

with torch.no_grad():
    try:
        x = load_keypoints_from_csv(str(seq_path)).cuda()
        # Rotation histogram
        hist = compute_rotation_histogram(x, bins_theta=8, bins_phi=4, L=15)
        # hist = compute_direction_histogram(x, bins_theta=8, bins_phi=4, L=15)
        # plot_histogram_single(hist, label_name=label_name, bins_theta=8, bins_phi=4)
        # dtheta dynamics
        mean_dtheta, cumulative_theta = compute_theta_dynamics_rotation(x)
    except Exception as e:
        print(f"[ERROR] {seq_path}: {e}")
# compute_theta_dynamics_rotation


