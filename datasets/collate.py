# """
# Inspired from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
# """
# import re
# import numpy as np
# import torch
# from torch.utils.data._utils.collate import default_collate
# from datasets.queries import BaseQueries, TransQueries 
# np_str_obj_array_pattern = re.compile(r"[SaUO]") 

# # def meshreg_collate(batch, extend_queries=None):
# #     """
# #     Collate function, duplicating the items in extend_queries along the
# #     first dimension so that they all have the same length.
# #     Typically applies to faces and vertices, which have different sizes
# #     depending on the object.
# #     """

# #     pop_queries = []
# #     # meshreg_collate 내부에서:
# #     if pop_query == BaseQueries.OBJIDX:
# #         continue  # skip auto-merge for obj_idx

# #     for poppable_query in extend_queries:
# #         if poppable_query in batch[0]:
# #             pop_queries.append(poppable_query)

# #     # Remove fields that don't have matching sizes
# #     for pop_query in pop_queries:
# #         max_size = max([sample[pop_query].shape[0] for sample in batch])
# #         for sample in batch:
# #             pop_value = sample[pop_query]
# #             # Repeat vertices so all have the same number
# #             pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
# #             sample[pop_query] = pop_value
# #     batch = default_collate(batch)
# #     return batch

# def meshreg_collate(batch, extend_queries=None):
#     """
#     Collate function for meshreg models. Handles variable-sized fields and multi-modal image tensors.
#     """
#     pop_queries = []
#     for poppable_query in extend_queries or []:
#         if poppable_query in [BaseQueries.OBJIDX, TransQueries.CAMINTR]:
#             continue
#         if poppable_query in batch[0]:
#             pop_queries.append(poppable_query)

#     for pop_query in pop_queries:
#         # ✅ Case 1: Multi-modal list (like [rgb, depth, thermal])
#         if isinstance(batch[0][pop_query], list) and isinstance(batch[0][pop_query][0], torch.Tensor):
#             num_modals = len(batch[0][pop_query])
#             modalwise_batches = [[] for _ in range(num_modals)]  # e.g., [[], [], []]

#             for sample in batch:
#                 for i in range(num_modals):
#                     modalwise_batches[i].append(sample[pop_query][i])

#             # Stack modal-wise
#             batch_modal = [torch.stack(modal, dim=0) for modal in modalwise_batches]
#             # 저장: batch[pop_query] = [B, C, H, W] * 3 형태 유지
#             for i, sample in enumerate(batch):
#                 sample[pop_query] = [modal[i] for modal in batch_modal]
        
#         # ✅ Case 2: 일반 텐서 (기존 방식)
#         elif isinstance(batch[0][pop_query], np.ndarray) or torch.is_tensor(batch[0][pop_query]):
#             max_size = max([sample[pop_query].shape[0] for sample in batch])
#             for sample in batch:
#                 pop_value = sample[pop_query]
#                 pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
#                 sample[pop_query] = pop_value

#     # OBJIDX는 리스트로 유지
#     obj_idx_list = [sample.pop(BaseQueries.OBJIDX, None) for sample in batch]

#     # 기본 collate
#     batch_collated = default_collate(batch)

#     # 다시 붙이기
#     batch_collated[BaseQueries.OBJIDX] = obj_idx_list

#     return batch_collated



# def seq_extend_flatten_collate(seq, extend_queries=None):
#     batch=[]    
#     seq_len = len(seq[0])#len(seq) is batch size, seq_len is num frames per sample

#     for sample in seq:
#         for seq_idx in range(seq_len):
#             batch.append(sample[seq_idx])
#     return meshreg_collate(batch,extend_queries)


import re
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from datasets.queries import BaseQueries, TransQueries 
np_str_obj_array_pattern = re.compile(r"[SaUO]") 

# def meshreg_collate(batch, extend_queries=None):
#     """
#     Collate function, duplicating the items in extend_queries along the
#     first dimension so that they all have the same length.
#     Typically applies to faces and vertices, which have different sizes
#     depending on the object.
#     """

#     pop_queries = []
#     # meshreg_collate 내부에서:
#     if pop_query == BaseQueries.OBJIDX:
#         continue  # skip auto-merge for obj_idx

#     for poppable_query in extend_queries:
#         if poppable_query in batch[0]:
#             pop_queries.append(poppable_query)

#     # Remove fields that don't have matching sizes
#     for pop_query in pop_queries:
#         max_size = max([sample[pop_query].shape[0] for sample in batch])
#         for sample in batch:
#             pop_value = sample[pop_query]
#             # Repeat vertices so all have the same number
#             pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
#             sample[pop_query] = pop_value
#     batch = default_collate(batch)
#     return batch

def meshreg_collate(batch, extend_queries=None):
    """
    Collate function for meshreg models. Handles variable-sized fields and multi-modal image tensors.
    """
    pop_queries = []
    for poppable_query in extend_queries or []:
        if poppable_query in [BaseQueries.OBJIDX, TransQueries.CAMINTR]:
            continue
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    for pop_query in pop_queries:
        # ✅ Case 1: Multi-modal list (like [rgb, depth, thermal])
        if isinstance(batch[0][pop_query], list) and isinstance(batch[0][pop_query][0], torch.Tensor):
            num_modals = len(batch[0][pop_query])
            modalwise_batches = [[] for _ in range(num_modals)]  # e.g., [[], [], []]

            for sample in batch:
                for i in range(num_modals):
                    modalwise_batches[i].append(sample[pop_query][i])

            # Stack modal-wise
            batch_modal = [torch.stack(modal, dim=0) for modal in modalwise_batches]
            # 저장: batch[pop_query] = [B, C, H, W] * 3 형태 유지
            for i, sample in enumerate(batch):
                sample[pop_query] = [modal[i] for modal in batch_modal]
        
        # ✅ Case 2: 일반 텐서 (기존 방식)
        elif isinstance(batch[0][pop_query], np.ndarray) or torch.is_tensor(batch[0][pop_query]):
            max_size = max([sample[pop_query].shape[0] for sample in batch])
            for sample in batch:
                pop_value = sample[pop_query]
                pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
                sample[pop_query] = pop_value

    # OBJIDX는 리스트로 유지
    obj_idx_list = [sample.pop(BaseQueries.OBJIDX, None) for sample in batch]

    # 기본 collate
    batch_collated = default_collate(batch)

    # 다시 붙이기
    batch_collated[BaseQueries.OBJIDX] = obj_idx_list

    return batch_collated



# def seq_extend_flatten_collate(seq, extend_queries=None):
#     batch=[]    
#     seq_len = len(seq[0])#len(seq) is batch size, seq_len is num frames per sample

#     # for sample in seq:
#     #     for seq_idx in range(seq_len):
#     #         batch.append(sample[seq_idx])
#     # return meshreg_collate(batch,extend_queries)

#     # collate.py - seq_extend_flatten_collate 내부
#     for sample in seq:
#         if isinstance(sample, list):  # HTT-style
#             sample = sample[seq_idx]
#         # 이제 sample은 dict임
#         batch.append(sample)
#     return meshreg_collate(batch,extend_queries)


#  ---new(handformer)
def seq_extend_flatten_collate(seq, extend_queries=None):
    # seq: [B, T] 구조의 list
    assert isinstance(seq[0], list), "각 sample은 list of dict 형태의 시퀀스여야 합니다."
    
    B = len(seq)
    T = len(seq[0])

    images, keypoints = [], []
    batch = []

    for sample_seq in seq:
        for frame_sample in sample_seq:
            if not isinstance(frame_sample, dict):
                raise ValueError(f"잘못된 sample 구조: {type(frame_sample)} → {frame_sample}")
            rgb_img, pose_kp = frame_sample[TransQueries.IMAGE]
            images.append(rgb_img)
            keypoints.append(pose_kp)
            del frame_sample[TransQueries.IMAGE]
            batch.append(frame_sample)

    batch_collated = meshreg_collate(batch, extend_queries)

    # keypoints [B, 3, T, J, M]
    keypoints_tensor = torch.stack(keypoints, dim=0)  # [B*T, 42, 3]
    keypoints_tensor = keypoints_tensor.view(B, T, 42, 3)

    # 21개씩 나누기 → left/right 손
    pose_left = keypoints_tensor[:, :, :21, :]  # [B, T, 21, 3]
    pose_right = keypoints_tensor[:, :, 21:, :] # [B, T, 21, 3]

    # 두 손 합치기 → [B, T, 21, 3, 2]
    pose_both = torch.stack([pose_left, pose_right], dim=-1)

    # 최종 shape → [B, 3, T, 21, 2]
    pose_both = pose_both.permute(0, 3, 1, 2, 4).contiguous()

    batch_collated["pose_keypoint"] = pose_both

    # RGB images [B, 3, T, H, W]
    rgb_tensor = torch.stack(images, dim=0)  # [B*T, 3, H, W]
    _, C, H, W = rgb_tensor.shape
    rgb_tensor = rgb_tensor.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    batch_collated["rgb_image"] = rgb_tensor

    return batch_collated
