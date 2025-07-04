# import traceback

# import numpy as np
# from PIL import Image, ImageFilter
# from torch.distributions.uniform import Uniform
# from torch.distributions.normal import Normal
# from torch.utils.data import Dataset
# from torchvision.transforms import functional as func_transforms

# from libyana.transformutils import colortrans, handutils
# from datasets.queries import BaseQueries, TransQueries, one_query_in 

# class SeqSet(Dataset): 
#     def __init__(
#         self,
#         pose_dataset,
#         inp_res,
#         scale_jittering,
#         center_jittering,
#         train,
#         queries,
#         ntokens,
#         spacing,
#         hue=0.15,
#         saturation=0.5,
#         contrast=0.5,
#         brightness=0.5,
#         blur_radius=0.5,
#     ):
#         # Dataset attributes
#         self.pose_dataset = pose_dataset
#         self.inp_res = tuple(inp_res) 
        
#         # Sequence attributes
#         self.ntokens = ntokens
#         self.spacing = spacing
        

#         # Color jitter attributes
#         self.hue = hue
#         self.contrast = contrast
#         self.brightness = brightness
#         self.saturation = saturation
#         self.blur_radius = blur_radius


#         # Training attributes
#         self.train = train
#         self.scale_jittering = scale_jittering
#         self.center_jittering = center_jittering


#         self.queries = queries



#     def __len__(self):
#         return len(self.pose_dataset)
    

#     # 수정(multi-modal)
#     def get_sample(self, idx, query=None, seq_txn=None, color_augm=None, space_augm=None):
#         if query is None:
#             query = self.queries
#         sample = {}

#         if BaseQueries.ACTIONIDX in query:
#             sample[BaseQueries.ACTIONIDX] = self.pose_dataset.get_action_idxs(idx)
#         if BaseQueries.OBJIDX in query:
#             sample[BaseQueries.OBJIDX] = self.pose_dataset.get_obj_idxs(idx)

#         center = np.array((480 / 2, 270 / 2))
#         scale = 480

#         # Data augmentation (spatial)
#         if space_augm is not None:
#             center = space_augm["center"]
#             scale = space_augm["scale"]
#         elif self.train:
#             center_jit = Uniform(-1, 1).sample((2,)).numpy()
#             center_offsets = self.center_jittering * scale * center_jit
#             center += center_offsets.astype(int)

#             scale_jit = 1 + Normal(0, 1).sample().item() * self.scale_jittering
#             scale_jit = np.clip(scale_jit, 1 - self.scale_jittering, 1 + self.scale_jittering)
#             scale *= scale_jit

#         sample["space_augm"] = {"scale": scale, "center": center}

#         # ===== Modal Images (RGB, Depth, Thermal) =====
#         # rgb, depth, thermal = self.pose_dataset.get_modal_images(idx, txn=seq_txn)
#         rgb, thermal = self.pose_dataset.get_modal_images(idx, txn=seq_txn)
#         mask = self.pose_dataset.get_mask(idx, txn=seq_txn)

#         # Data augmentation (blur)
#         if self.train:
#             blur_radius = Uniform(0, 1).sample().item() * self.blur_radius
#             rgb = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
#             # depth = depth.filter(ImageFilter.GaussianBlur(blur_radius))
#             # thermal = thermal.filter(ImageFilter.GaussianBlur(blur_radius))

#         # Data augmentation (color jittering)
#         if self.train:
#             if color_augm is None:
#                 bright, contrast, sat, hue = colortrans.get_color_params(
#                     brightness=self.brightness,
#                     saturation=self.saturation,
#                     hue=self.hue,
#                     contrast=self.contrast,
#                 )
#             else:
#                 sat = color_augm["sat"]
#                 contrast = color_augm["contrast"]
#                 hue = color_augm["hue"]
#                 bright = color_augm["bright"]

#             rgb = colortrans.apply_jitter(rgb, brightness=bright, saturation=sat, hue=hue, contrast=contrast)
#             # depth와 thermal은 grayscale로 hue/sat X
#             sample["color_augm"] = {"sat": sat, "bright": bright, "contrast": contrast, "hue": hue}
#         else:
#             sample["color_augm"] = None

#         # Create buffer white image if needed
#         # Transform and crop
#         img = handutils.transform_img(rgb, affinetrans, self.inp_res)
#         mask = handutils.transform_img(mask, affinetrans, self.inp_res)
#         img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))
#         mask = mask.crop((0, 0, self.inp_res[0], self.inp_res[1]))

#         # Tensorize and normalize_img
#         img = func_transforms.to_tensor(img).float()
#         img = func_transforms.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
#         mask = func_transforms.to_tensor(mask).float()
#         # mask = func_transforms.normalize(mask, [0.5, 0.5, 0.5], [1, 1, 1])
#         if TransQueries.IMAGE in query:
#             sample[TransQueries.IMAGE] = img
#             sample[TransQueries.MASK] = mask
#         if TransQueries.IMAGE_T in query:
#             img_t = handutils.transform_img(img_t, affinetrans, self.inp_res)
#             img_t = func_transforms.to_tensor(img_t).float()
#             img_t = func_transforms.normalize(img_t, [0.5, 0.5, 0.5], [1, 1, 1])
#             sample[TransQueries.IMAGE_T] = img_t

#         # Normalize & to_tensor
#         def normalize_img(im):
#             im_tensor = func_transforms.to_tensor(im).float()
#             if im_tensor.shape[0] == 1:
#                 return func_transforms.normalize(im_tensor, [0.5], [1.0])
#             else:
#                 return func_transforms.normalize(im_tensor, [0.5, 0.5, 0.5], [1, 1, 1])

#         rgb_tensor = normalize_img(rgb)
#         # depth_tensor = normalize_img(depth)
#         thermal_tensor = normalize_img(thermal)

#         # sample[TransQueries.IMAGE] = [rgb_tensor, depth_tensor, thermal_tensor]
#         sample[TransQueries.IMAGE] = [rgb_tensor,thermal_tensor]
        

#         # ===== Hand label =====
#         hand_label = self.pose_dataset.get_hand_label(idx)
#         if hand_label is None:
#             hand_label = [0, 0]
#         sample["hand_label_left"] = hand_label[0]
#         sample["hand_label_right"] = hand_label[1]

#         # ===== Meta Info =====
#         sample["sample_info"] = self.pose_dataset.get_sample_info(idx)
#         return sample


#         def get_safesample(self, idx,seq_txn=None, color_augm=None, space_augm=None):
#             try:
#                 sample = self.get_sample(idx, self.queries, seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
#             except Exception:
#                 traceback.print_exc()
#                 assert False, f"Encountered error processing sample {idx}" 
#             return sample

#     def __getitem__(self, idx, verbose=False):
#         fidx=self.pose_dataset.get_start_frame_idx(idx)
#         seq_txn=self.pose_dataset.open_seq_lmdb(fidx)

#         sample = self.get_sample(fidx,seq_txn=seq_txn)
#         frame_idx=self.pose_dataset.get_dataidx(fidx)

#         sample["dist2query"] = 0
#         sample["not_padding"] = 1
#         space_augm = sample.pop("space_augm")
#         color_augm = sample.pop("color_augm")

#         samples = [sample]  
#         cur_idx=frame_idx
#         for sample_idx in range(self.ntokens-1):
#             fut_idx, fut_not_padding=self.pose_dataset.get_future_frame_idx(cur_idx=cur_idx,
#                                                 fut_idx=cur_idx+self.spacing,
#                                                 spacing=self.spacing,
#                                                 verbose=verbose)
#             sample_fut_frame = self.get_sample(fut_idx,seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
#             sample_fut_frame["dist2query"] = fut_idx-frame_idx
#             sample_fut_frame["not_padding"] = fut_not_padding

#             sample_fut_frame.pop("space_augm")
#             sample_fut_frame.pop("color_augm")
#             samples.append(sample_fut_frame)

#             if fut_idx!=cur_idx:
#                 cur_idx=fut_idx
#         return samples




import traceback

import numpy as np
from PIL import Image, ImageFilter
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms

from libyana.transformutils import colortrans, handutils
from datasets.queries import BaseQueries, TransQueries, one_query_in
import torch
class SeqSet(Dataset): 
    def __init__(
        self,
        pose_dataset,
        inp_res,
        scale_jittering,
        center_jittering,
        train,
        queries,
        ntokens,
        spacing,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
    ):
        self.pose_dataset = pose_dataset
        self.inp_res = tuple(inp_res)
        self.ntokens = ntokens
        self.spacing = spacing

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries

    def __len__(self):
        return len(self.pose_dataset)

    def get_sample(self, idx, query=None, seq_txn=None, color_augm=None, space_augm=None):
        if query is None:
            query = self.queries
        sample = {}

        if BaseQueries.ACTIONIDX in query:
            sample[BaseQueries.ACTIONIDX] = self.pose_dataset.get_action_idxs(idx)
        if BaseQueries.OBJIDX in query:
            sample[BaseQueries.OBJIDX] = self.pose_dataset.get_obj_idxs(idx)

        center = np.array((480 / 2, 270 / 2))
        scale = 480

        if space_augm is not None:
            center = space_augm["center"]
            scale = space_augm["scale"]
        elif self.train:
            center_jit = Uniform(-1, 1).sample((2,)).numpy()
            center_offsets = self.center_jittering * scale * center_jit
            center += center_offsets.astype(int)

            scale_jit = 1 + Normal(0, 1).sample().item() * self.scale_jittering
            scale_jit = np.clip(scale_jit, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale *= scale_jit

        sample["space_augm"] = {"scale": scale, "center": center}

        # ===== Modal Images (RGB, Thermal) =====
        rgb, thermal = self.pose_dataset.get_modal_images(idx, txn=seq_txn)
        # mask = self.pose_dataset.get_mask(idx, txn=seq_txn)

        if self.train:
            blur_radius = Uniform(0, 1).sample().item() * self.blur_radius
            rgb = rgb.filter(ImageFilter.GaussianBlur(blur_radius))

        if self.train:
            if color_augm is None:
                bright, contrast, sat, hue = colortrans.get_color_params(
                    brightness=self.brightness,
                    saturation=self.saturation,
                    hue=self.hue,
                    contrast=self.contrast,
                )
            else:
                sat = color_augm["sat"]
                contrast = color_augm["contrast"]
                hue = color_augm["hue"]
                bright = color_augm["bright"]

            rgb = colortrans.apply_jitter(rgb, brightness=bright, saturation=sat, hue=hue, contrast=contrast)
            sample["color_augm"] = {"sat": sat, "bright": bright, "contrast": contrast, "hue": hue}
        else:
            sample["color_augm"] = None

        # Affine transform 준비
        affinetrans, _ = handutils.get_affine_transform(center, scale, self.inp_res, rot=0)

        # (RGB) Transform, crop, to_tensor, normalize
        img = handutils.transform_img(rgb, affinetrans, self.inp_res)
        img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        img = func_transforms.to_tensor(img).float()
        img = func_transforms.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])

        # (Thermal) Transform, crop, to_tensor, normalize
        # thermal = handutils.transform_img(thermal, affinetrans, self.inp_res)
        # thermal = thermal.crop((0, 0, self.inp_res[0], self.inp_res[1]))
        # thermal = func_transforms.to_tensor(thermal).float()
        # thermal = func_transforms.normalize(thermal, [0.5], [1.0])

        # 기존 thermal 처리 대신 CSV 경로 로딩
        try:
            keypoint_np = np.loadtxt(thermal, delimiter=",", skiprows=1)  # shape: [N, 3]
            if keypoint_np.shape[0] != 42:
                keypoint_np = np.pad(keypoint_np, ((0, 42 - keypoint_np.shape[0]), (0, 0)), mode="constant")
        except Exception:
            keypoint_np = np.zeros((42, 3), dtype=np.float32)

        keypoint_tensor = torch.tensor(keypoint_np, dtype=torch.float32)  # shape: [42, 3]

        # Save into sample dict
        if TransQueries.IMAGE in query:
            sample[TransQueries.IMAGE] = img
            # sample[TransQueries.MASK] = mask
        if TransQueries.IMAGE in query:
            sample[TransQueries.IMAGE] =  keypoint_tensor

        # [Optional] 둘을 list로 같이 묶어서 주는 경우
        sample[TransQueries.IMAGE] = [img,  keypoint_tensor]

        # ===== Hand label =====
        hand_label = self.pose_dataset.get_hand_label(idx)
        import pdb;pdb.set_trace()
        if hand_label is None:
            hand_label = [0, 0]
        sample["hand_label_left"] = hand_label[0]
        sample["hand_label_right"] = hand_label[1]

        # ===== Meta Info =====
        sample["sample_info"] = self.pose_dataset.get_sample_info(idx)
        return sample

    def get_safesample(self, idx, seq_txn=None, color_augm=None, space_augm=None):
        try:
            sample = self.get_sample(idx, self.queries, seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
        except Exception:
            traceback.print_exc()
            assert False, f"Encountered error processing sample {idx}" 
        return sample

    def __getitem__(self, idx, verbose=False):
        fidx = self.pose_dataset.get_start_frame_idx(idx)
        seq_txn = self.pose_dataset.open_seq_lmdb(fidx)

        sample = self.get_sample(fidx, seq_txn=seq_txn)
        frame_idx = self.pose_dataset.get_dataidx(fidx)

        sample["dist2query"] = 0
        sample["not_padding"] = 1
        space_augm = sample.pop("space_augm")
        color_augm = sample.pop("color_augm")

        samples = [sample]
        cur_idx = frame_idx
        for sample_idx in range(self.ntokens - 1):
            fut_idx, fut_not_padding = self.pose_dataset.get_future_frame_idx(
                cur_idx=cur_idx,
                fut_idx=cur_idx + self.spacing,
                spacing=self.spacing,
                verbose=verbose
            )
            sample_fut_frame = self.get_sample(fut_idx, seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
            sample_fut_frame["dist2query"] = fut_idx - frame_idx
            sample_fut_frame["not_padding"] = fut_not_padding

            sample_fut_frame.pop("space_augm")
            sample_fut_frame.pop("color_augm")
            samples.append(sample_fut_frame)

            if fut_idx != cur_idx:
                cur_idx = fut_idx
        return samples


