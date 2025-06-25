# # #-----0613
# # import torch
# # import pandas as pd
# # import numpy as np
# # import torch.nn as nn
# # from pathlib import Path
# # from HandFormer.HandFormer.models.ms_tcn_1D import MultiScale_TemporalConv as MS_TCN
# # import torch.nn.functional as F
# # import matplotlib.pyplot as plt
# # from sklearn.manifold import TSNE
# # from matplotlib.patches import Patch
# # import os

# # def normalize_tensor(tensor, dim=1):
# #     norm = torch.norm(tensor, dim=dim, keepdim=True).clamp(min=1e-12)
# #     return tensor / norm

# # def load_keypoints_from_csv(csv_folder, max_frames=500):
# #     csv_paths = sorted(Path(csv_folder).glob("*.csv"))[:max_frames]
# #     frames = []
# #     for csv_path in csv_paths:
# #         df = pd.read_csv(csv_path)

# #         # 양손(42줄)이 아니면 skip
# #         if df.shape[0] != 42:
# #             continue

# #         # 모든 값이 0이면 skip (손 없음)
# #         if df[['x', 'y', 'z']].abs().sum().sum() == 0:
# #             continue

# #         frame = df[['x', 'y', 'z']].values.astype(np.float32).reshape(2, 21, 3)  # (M, V, C)
# #         frames.append(frame)

# #     if not frames:
# #         raise ValueError("No valid CSV keypoints found.")
    
# #     kp = np.stack(frames, axis=0)                  # (T, M, V, C)
# #     kp = np.transpose(kp, (3, 0, 2, 1))             # (C, T, V, M)
# #     kp = np.expand_dims(kp, axis=0)                # (1, C, T, V, M)
# #     return torch.tensor(kp, dtype=torch.float32)



# # def compute_palm_normal(x):
# #     idx = [5, 9, 13]
# #     p0, p1, p2 = x[:, :, :, idx[0], :], x[:, :, :, idx[1], :], x[:, :, :, idx[2], :]
# #     v1 = p1 - p0
# #     v2 = p2 - p0
# #     normal = torch.cross(v1, v2, dim=1)
# #     normal = normalize_tensor(normal, dim=1)
# #     return normal


# # def smooth_normal(normal, kernel_size=5):
# #     pad = kernel_size //2
# #     normal_ = normal.permute(0,3,1,2).reshape(-1, normal.shape[1], normal.shape[2])
# #     smooth = F.avg_pool1d(normal_, kernel_size=kernel_size, stride=1, padding=pad)
# #     smooth = smooth.view(normal.shape[0], normal.shape[3], normal.shape[1], normal.shape[2])
# #     smooth = smooth.permute(0,2,3,1)
# #     smooth = normalize_tensor(smooth, dim=1)
# #     return smooth

# # class RotationTCN(nn.Module):
# #     def __init__(self, token_dim=256):
# #         super().__init__()
# #         self.tcn = nn.Sequential(
# #             MS_TCN(1, 64),
# #             MS_TCN(64, 128),
# #             MS_TCN(128, token_dim),
# #         )

# #     def forward(self, angle_seq):
# #         N, T, M = angle_seq.shape
# #         x = angle_seq.permute(0, 2, 1).contiguous().view(N * M, 1, T)
# #         out = self.tcn(x).mean(dim=-1)
# #         return out.view(N, M, -1).mean(dim=1)


# # class DirectionTokenExtractor(nn.Module):
# #     def __init__(self, token_dim=256):
# #         super().__init__()
# #         self.mlp = nn.Sequential(
# #             nn.Linear(1, token_dim),
# #             nn.ReLU(),
# #             nn.Linear(token_dim, token_dim)
# #         )

# #     def forward(self, signed_angles):
# #         agg = signed_angles.mean(dim=-1, keepdim=True)
# #         token = self.mlp(agg.unsqueeze(-1)).squeeze(-1)
# #         return token.mean(dim=1)

# # # def compute_directional_token(x, token_dim=256):
# # #     B, C, T, V, M = x.shape
# # #     palm_normal = compute_palm_normal(x)  # (B, C, T, M)
# # #     palm_normal = smooth_normal(palm_normal, kernel_size=5)
# # #     # import pdb;pdb.set_trace()
# # #     palm_t = palm_normal[:, :, :-1, :]
# # #     palm_tp1 = palm_normal[:, :, 1:, :]
# # #     dot = (palm_tp1 * palm_t).sum(dim=1).clamp(-1.0, 1.0)
# # #     angle_seq = torch.acos(dot)
# # #     palm_dynamics = RotationTCN(token_dim).cuda()(angle_seq)
# # #     consistency = (angle_seq > 0).float().mean(dim=1, keepdim=True)
# # #     direction_token = DirectionTokenExtractor(token_dim).cuda()(consistency)
# # #     token = palm_dynamics + direction_token
# # #     return token


# # def compute_directional_token(x, token_dim=256):
# #     B, C, T, V, M = x.shape
# #     palm_normal = compute_palm_normal(x)
# #     palm_normal = smooth_normal(palm_normal, kernel_size=5)  # smoothing 추가
# #     palm_t = palm_normal[:, :, :-1, :]
# #     palm_tp1 = palm_normal[:, :, 1:, :]
# #     dot_sim = (palm_tp1 * palm_t).sum(dim=1).clamp(-1.0, 1.0)
# #     stability = dot_sim.mean(dim=1, keepdim=True)
# #     angle_seq = torch.acos(dot_sim)

# #     # stability-weighted direction token
# #     direction_token_raw = DirectionTokenExtractor(token_dim).cuda()(angle_seq)
# #     direction_token = direction_token_raw * stability.mean(dim=-1)

# #     # rotation direction cue
# #     cross = torch.cross(palm_t, palm_tp1, dim=1)
# #     angle = torch.atan2(cross.norm(dim=1), dot_sim)
# #     sign = torch.sign(cross[:, 2, :, :])
# #     signed_angle = angle * sign
# #     cumsum_angle = signed_angle.cumsum(dim=1)
# #     rotation_token = RotationTCN(token_dim).cuda()(cumsum_angle)
# #     final_token = direction_token + rotation_token
# #     return cumsum_angle


# # # ===== Sequence list from Excel =====
# # df = pd.read_excel("/data/jyseo/functional_hand_type/0612_screw_unscrew_filter.xlsx")
# # valid_rows = df[df.iloc[:, 1] == 'O']
# # sequence_names = valid_rows.iloc[:, 0].tolist()

# # # ===== Gather data paths =====
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


# # # data_paths = {
# # #     "screw": sorted(base_dir.glob("screw/*/*/pose_2d_mp_csv")),
# # #     "unscrew": sorted(base_dir.glob("unscrew/*/*/pose_2d_mp_csv")),
# # # }



# # # ===== Compute tokens and save =====
# # save_dir = Path("./directional_tokens")
# # save_dir.mkdir(exist_ok=True)

# # with torch.no_grad():
# #     for cls, paths in data_paths.items():
# #         for path in paths:
# #             try:
# #                 x = load_keypoints_from_csv(str(path)).cuda()
# #                 token = compute_directional_token(x).squeeze(0).cpu().numpy()
# #                 name = path.parent.name
# #                 np.save(save_dir / f"{name}.npy", token)
# #             except:
# #                 continue


# # # ===== t-SNE Plotting =====
# # tokens, labels, label_names = [], [], []
# # for file in os.listdir(save_dir):
# #     if not file.endswith(".npy"):
# #         continue
# #     if "unscrew" in file:
# #         label = 1
# #         name = "unscrew"
# #     elif "screw" in file:
# #         label = 0
# #         name = "screw"
# #     else:
# #         continue
# #     token = np.load(save_dir / file).reshape(-1)  # ✅ shape 강제 보정
# #     tokens.append(token)
# #     labels.append(label)
# #     label_names.append(name)

# # tokens = np.stack(tokens)
# # labels = np.array(labels)
# # tsne = TSNE(n_components=2, random_state=42, perplexity=3)
# # tokens_2d = tsne.fit_transform(tokens)

# # plt.figure(figsize=(10, 8))
# # colors = ['blue' if l == 0 else 'red' for l in labels]
# # plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=colors, s=50, alpha=0.7)
# # for i, label in enumerate(label_names):
# #     plt.text(tokens_2d[i, 0] + 1.0, tokens_2d[i, 1], label, fontsize=9)
# # plt.legend(handles=[Patch(facecolor='blue', label='screw'), Patch(facecolor='red', label='unscrew')])
# # plt.title("TSNE of Directional Tokens (Palm Plane Normal)")
# # plt.xlabel("Dim 1")
# # plt.ylabel("Dim 2")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("tsne_directional_token_palmnormal.png", dpi=300)import torch

# import torch
# import torch
# import pandas as pd
# import numpy as np
# import torch.nn.functional as F
# from pathlib import Path
# from sklearn.manifold import TSNE
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import os

# def normalize_tensor(tensor, dim=1):
#     norm = torch.norm(tensor, dim=dim, keepdim=True).clamp(min=1e-12)
#     return tensor / norm

# def load_keypoints_from_csv(csv_folder, max_frames=None):
#     csv_paths = sorted(Path(csv_folder).glob("*.csv"))
#     if max_frames is not None:
#         csv_paths = csv_paths[:max_frames]
#     frames = []
#     for csv_path in csv_paths:
#         df = pd.read_csv(csv_path)
#         if df.shape[0] != 42:
#             continue
#         if df[['x', 'y', 'z']].abs().sum().sum() == 0:
#             continue
#         frame = df[['x', 'y', 'z']].values.astype(np.float32).reshape(2, 21, 3)
#         frames.append(frame)
#     if not frames:
#         raise ValueError("No valid CSV keypoints found.")
#     kp = np.stack(frames, axis=0)
#     kp = np.transpose(kp, (3, 0, 2, 1))  # (C, T, V, M)
#     kp = np.expand_dims(kp, axis=0)      # (1, C, T, V, M)
#     return torch.tensor(kp, dtype=torch.float32)

# def compute_palm_normal(x):
#     idx = [5, 9, 13]
#     p0, p1, p2 = x[:, :, :, idx[0], :], x[:, :, :, idx[1], :], x[:, :, :, idx[2], :]
#     v1 = p1 - p0
#     v2 = p2 - p0
#     normal = torch.cross(v1, v2, dim=1)
#     normal = normalize_tensor(normal, dim=1)
#     return normal

# def smooth_normal(normal, kernel_size=5):
#     pad = kernel_size // 2
#     normal_ = normal.permute(0, 3, 1, 2).reshape(-1, normal.shape[1], normal.shape[2])
#     smooth = F.avg_pool1d(normal_, kernel_size=kernel_size, stride=1, padding=pad)
#     smooth = smooth.view(normal.shape[0], normal.shape[3], normal.shape[1], normal.shape[2])
#     smooth = smooth.permute(0, 2, 3, 1)
#     smooth = normalize_tensor(smooth, dim=1)
#     return smooth

# def compute_joint_palm_combined(x):
#     B, C, T, V, M = x.shape
#     palm_normal = compute_palm_normal(x)
#     palm_normal = smooth_normal(palm_normal, kernel_size=5)
#     palm_t = palm_normal[:, :, :-1, :]
#     palm_tp1 = palm_normal[:, :, 1:, :]
#     dot_palm = (palm_tp1 * palm_t).sum(dim=1).clamp(-1, 1)
#     cross_palm = torch.cross(palm_t, palm_tp1, dim=1)
#     angle_palm = torch.atan2(cross_palm.norm(dim=1), dot_palm)
#     sign_palm = torch.sign(cross_palm[:, 2, :, :])
#     signed_angle_palm = angle_palm * sign_palm
#     cumsum_palm = signed_angle_palm.cumsum(dim=1)

#     joint_t = x[:, :, :-1, [0, 9], :]
#     joint_tp1 = x[:, :, 1:, [0, 9], :]
#     vec_t = normalize_tensor(joint_t[:, :, :, 1, :] - joint_t[:, :, :, 0, :], dim=1)
#     vec_tp1 = normalize_tensor(joint_tp1[:, :, :, 1, :] - joint_tp1[:, :, :, 0, :], dim=1)
#     dot_joint = (vec_tp1 * vec_t).sum(dim=1).clamp(-1, 1)
#     cross_joint = torch.cross(vec_t, vec_tp1, dim=1)
#     angle_joint = torch.atan2(cross_joint.norm(dim=1), dot_joint)
#     sign_joint = torch.sign(cross_joint[:, 2, :, :])
#     signed_angle_joint = angle_joint * sign_joint
#     cumsum_joint = signed_angle_joint.cumsum(dim=1)

#     combined = torch.cat([
#         cumsum_palm.view(B, -1),
#         cumsum_joint.view(B, -1)
#     ], dim=1)
#     return combined

# # ===== 데이터 로드 =====
# df = pd.read_excel("/data/jyseo/functional_hand_type/0612_screw_unscrew_filter.xlsx")
# valid_rows = df[df.iloc[:, 1] == 'O']
# sequence_names = valid_rows.iloc[:, 0].tolist()
# base_dir = Path("../data_MHAV")
# data_paths = {"screw": [], "unscrew": []}
# for seq_name in sequence_names:
#     if seq_name.startswith("screw"):
#         label = "screw"
#     elif seq_name.startswith("unscrew"):
#         label = "unscrew"
#     else:
#         continue
#     try:
#         hand_type = seq_name.split("_")[-1]
#         seq_path = base_dir / label / hand_type / seq_name / "pose_2d_mp_csv"
#         if seq_path.exists():
#             data_paths[label].append(seq_path)
#     except:
#         continue

# # ===== feature 추출 =====
# tokens = []
# labels = []
# with torch.no_grad():
#     for cls, paths in data_paths.items():
#         for path in paths:
#             try:
#                 x = load_keypoints_from_csv(str(path)).cuda()
#                 combined = compute_joint_palm_combined(x).squeeze(0).cpu().numpy()
#                 tokens.append(combined)
#                 labels.append(0 if cls == "screw" else 1)
#             except Exception as e:
#                 print(f"[ERROR] {path}: {e}")
#                 continue

# # ===== zero padding =====
# max_len = max(t.shape[0] for t in tokens)
# tokens_padded = []
# for t in tokens:
#     if t.shape[0] < max_len:
#         t = np.pad(t, (0, max_len - t.shape[0]))
#     tokens_padded.append(t)
# tokens = np.stack(tokens_padded)
# labels = np.array(labels)

# # ===== separability 검사 =====
# scaler = StandardScaler()
# tokens_scaled = scaler.fit_transform(tokens)

# logreg = LogisticRegression(max_iter=1000)
# logreg_scores = cross_val_score(logreg, tokens_scaled, labels, cv=5)
# print(f"Logistic Regression CV Accuracy: {logreg_scores.mean():.4f} ± {logreg_scores.std():.4f}")

# svm_lin = SVC(kernel="linear")
# svm_lin_scores = cross_val_score(svm_lin, tokens_scaled, labels, cv=5)
# print(f"SVM (linear) CV Accuracy: {svm_lin_scores.mean():.4f} ± {svm_lin_scores.std():.4f}")

# svm_rbf = SVC(kernel="rbf")
# svm_rbf_scores = cross_val_score(svm_rbf, tokens_scaled, labels, cv=5)
# print(f"SVM (RBF) CV Accuracy: {svm_rbf_scores.mean():.4f} ± {svm_rbf_scores.std():.4f}")

# # ===== t-SNE =====
# tsne = TSNE(n_components=2, random_state=42, perplexity=3)
# tokens_2d = tsne.fit_transform(tokens_scaled)

# plt.figure(figsize=(10,8))
# colors = ['blue' if l == 0 else 'red' for l in labels]
# plt.scatter(tokens_2d[:,0], tokens_2d[:,1], c=colors, s=50, alpha=0.7)
# plt.legend(handles=[
#     Patch(facecolor='blue', label='screw'),
#     Patch(facecolor='red', label='unscrew')
# ])
# plt.title("TSNE of Combined Palm + Joint Dynamics")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("tsne_joint_palm_combined.png", dpi=300)
# print("Saved t-SNE plot to tsne_joint_palm_combined.png")


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    kp = np.transpose(kp, (3, 0, 2, 1))
    kp = np.expand_dims(kp, axis=0)
    return torch.tensor(kp, dtype=torch.float32)

def compute_joint_palm_combined(x):
    B, C, T, V, M = x.shape
    # Palm dynamics
    idx = [5, 9, 13]
    p0, p1, p2 = x[:, :, :, idx[0], :], x[:, :, :, idx[1], :], x[:, :, :, idx[2], :]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = torch.cross(v1, v2, dim=1)
    normal = normalize_tensor(normal, dim=1)
    normal_t = normal[:, :, :-1, :]
    normal_tp1 = normal[:, :, 1:, :]
    dot_palm = (normal_tp1 * normal_t).sum(dim=1).clamp(-1, 1)
    cross_palm = torch.cross(normal_t, normal_tp1, dim=1)
    angle_palm = torch.atan2(cross_palm.norm(dim=1), dot_palm)
    sign_palm = torch.sign(cross_palm[:, 2, :, :])
    signed_angle_palm = angle_palm * sign_palm
    cumsum_palm = signed_angle_palm.cumsum(dim=1)

    # Joint dynamics
    joint_t = x[:, :, :-1, [0, 9], :]
    joint_tp1 = x[:, :, 1:, [0, 9], :]
    vec_t = normalize_tensor(joint_t[:, :, :, 1, :] - joint_t[:, :, :, 0, :], dim=1)
    vec_tp1 = normalize_tensor(joint_tp1[:, :, :, 1, :] - joint_tp1[:, :, :, 0, :], dim=1)
    dot_joint = (vec_tp1 * vec_t).sum(dim=1).clamp(-1, 1)
    cross_joint = torch.cross(vec_t, vec_tp1, dim=1)
    angle_joint = torch.atan2(cross_joint.norm(dim=1), dot_joint)
    sign_joint = torch.sign(cross_joint[:, 2, :, :])
    signed_angle_joint = angle_joint * sign_joint
    cumsum_joint = signed_angle_joint.cumsum(dim=1)

    combined = torch.cat([
        cumsum_palm.view(B, -1),
        cumsum_joint.view(B, -1)
    ], dim=1)
    return combined

# 데이터 로드 및 feature 추출
df = pd.read_excel("/data/jyseo/functional_hand_type/0612_screw_unscrew_filter.xlsx")
valid_rows = df[df.iloc[:, 1] == 'O']
sequence_names = valid_rows.iloc[:, 0].tolist()
base_dir = Path("../data_MHAV")
data_paths = {"screw": [], "unscrew": []}
for seq_name in sequence_names:
    if seq_name.startswith("screw"):
        label = "screw"
    elif seq_name.startswith("unscrew"):
        label = "unscrew"
    else:
        continue
    try:
        hand_type = seq_name.split("_")[-1]
        seq_path = base_dir / label / hand_type / seq_name / "pose_2d_mp_csv"
        if seq_path.exists():
            data_paths[label].append(seq_path)
    except:
        continue

tokens = []
labels = []
with torch.no_grad():
    for cls, paths in data_paths.items():
        for path in paths:
            try:
                x = load_keypoints_from_csv(str(path)).cuda()
                feat = compute_joint_palm_combined(x).squeeze(0).cpu().numpy()
                tokens.append(feat)
                labels.append(0 if cls == "screw" else 1)
            except Exception as e:
                print(f"[ERROR] {path}: {e}")
                continue

# Padding
max_len = max(t.shape[0] for t in tokens)
tokens_padded = [np.pad(t, (0, max_len - t.shape[0])) for t in tokens]
tokens = np.stack(tokens_padded)
labels = np.array(labels)

# Contrastive loss 계산
tokens_tensor = torch.tensor(tokens, dtype=torch.float32)
scaler = StandardScaler()
tokens_scaled = scaler.fit_transform(tokens_tensor)
tokens_scaled = torch.tensor(tokens_scaled, dtype=torch.float32)

# 쌍 생성
pairs = []
targets = []
N = tokens_scaled.shape[0]
for i in range(N):
    for j in range(i + 1, N):
        pairs.append((tokens_scaled[i], tokens_scaled[j]))
        targets.append(1 if labels[i] == labels[j] else -1)

x1 = torch.stack([p[0] for p in pairs])
x2 = torch.stack([p[1] for p in pairs])
targets = torch.tensor(targets, dtype=torch.float32)

proj = torch.nn.Linear(tokens_scaled.shape[1], 128)
out1 = F.normalize(proj(x1), dim=-1)
out2 = F.normalize(proj(x2), dim=-1)
criterion = torch.nn.CosineEmbeddingLoss()
loss = criterion(out1, out2, targets)
print(f"Contrastive loss: {loss.item():.4f}")

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
tokens_2d = tsne.fit_transform(tokens_scaled.numpy())
plt.figure(figsize=(10,8))
colors = ['blue' if l == 0 else 'red' for l in labels]
plt.scatter(tokens_2d[:,0], tokens_2d[:,1], c=colors, s=50, alpha=0.7)
plt.legend(handles=[
    Patch(facecolor='blue', label='screw'),
    Patch(facecolor='red', label='unscrew')
])
plt.title("TSNE of Combined Palm + Joint Dynamics")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_joint_palm_combined.png", dpi=300)
print("Saved t-SNE plot to tsne_joint_palm_combined.png")
