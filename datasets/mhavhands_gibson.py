
from os import path
import os
import sys
import torch
sys.path.append(path.abspath('.'))

import lmdb

import numpy as np
from PIL import Image, ImageFile
from datasets import mhavutils_gibson
from datasets.queries import BaseQueries,TransQueries, get_trans_queries
from datasets import mhavutils_gibson
from config import DATA_ROOT_PATH


ImageFile.LOAD_TRUNCATED_IMAGES = True


class MHAVhands(object):
    def __init__(
        self,
        dataset_folder,
        split,#
        ntokens_pose,
        ntokens_action,
        spacing,
        is_shifting_window,
        split_type="actions",
    ):
        super().__init__()

        self.ntokens_pose = ntokens_pose
        self.ntokens_action=ntokens_action
        self.spacing=spacing
        self.is_shifting_window=is_shifting_window

        self.all_queries = [
            BaseQueries.IMAGE,           
            BaseQueries.CAMINTR,

            TransQueries.JOINTS2D, 
            TransQueries.JOINTSABS25D,
            TransQueries.CAMINTR,

            BaseQueries.JOINTS3D,
            BaseQueries.ACTIONIDX,
            BaseQueries.OBJIDX,
        ]
             
       
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries) 

        # Get camera info
        self.cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]])

        self.reorder_idx = np.array(
            [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
        )
        self.name = "mhav"
        split_opts = ["actions", "subjects", "handtypes"] # action, subject, handtype, tool
        self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5"] #5
        
        if split_type not in split_opts:
            raise ValueError(
                "Split for dataset {} should be in {}, got {}".format(self.name, split_opts, split_type)
            )
        self.split = split
        self.split_type = split_type 
        

        self.root = "../data_MHAV"
        
        self.info_root = os.path.join(self.root, "Subjects_info")

        self.reduce_res = True


        # # self.split = split
        self.rgb_template = "RGB_undistorted_{}.jpg"
        print("앙 확인띠", self.root)
        small_rgb = self.root  # 또는 실제 리사이즈 이미지 경로

        if os.path.exists(small_rgb) and self.reduce_res:
            self.rgb_root = small_rgb
            self.reduce_factor = 1 / 4
            print("✅ Using reduced resolution.")
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")  # 🔧 수정
            self.reduce_factor = 1
            print("⚠️ Warning: reduce factor is 1. Using full-res images.")


        #Load action labels
        path_action_info = '../data_MHAV/action_info.txt'
        action_info, action_to_idx = mhavutils_gibson.get_action_infos(path_action_info)
        self.action_info=action_info
        self.action_to_idx=action_to_idx
        self.num_actions = len(self.action_info.keys())

        #Load object labels
        path_object_info = '../data_MHAV/object_tool_id.txt'
        object_info, object_to_idx = mhavutils_gibson.get_object_infos(path_object_info)
        self.object_info=object_info
        self.object_to_idx=object_to_idx
        self.num_objects = len(self.object_info.keys())


        # Load hand type labels
        self.num_handtypes = 39
        self.hand_labels, self.subjects_infos = mhavutils_gibson.get_all_hand_labels('../data_MHAV/annotation_0620_final.xlsx', self.rgb_root, self.rgb_template)
        # print("self.hand_labels keys:", list(self.hand_labels))  # 현재 hand_labels에 어떤 키들이 있는지 확인
        # print("self.subjects_labels keys:", list(self.subjects_infos))
        self.mhav_hand_map={-1: 0,  1: 1,  2: 2,  3: 3,  4: 4,  8: 5,  9: 6,  12: 7,  14: 8,  15: 9,  17: 10,  18: 11,  22: 12,  23: 13,  27: 14, 29: 15,  32: 16,  33: 17,  34: 18,  36: 19,  38: 20,  39: 21,  40: 22,  41: 23,  42: 24,  43: 25,  44: 26,  45: 27,  46: 28,  47: 29,  48: 30,  49: 31,  50: 32,  51: 33,  52: 34,  53: 35,  54: 36,  55: 37,  56: 38}

        
        for i,(k,v) in enumerate(self.action_info.items()):
            self.action_info[k]["action_idx"]=i
        

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
        self.load_dataset()

        # Infor for rendering
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.image_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)] 


        self.env_r=None
                
    
    def load_dataset(self):
        if self.split_type == "subjects":
            if self.split == "train":
                subjects = ['jl', 'sp', 'kl', 'pc']
            elif self.split in ["test", "val"]:
                subjects = ['hg']
            else:
                raise ValueError(f"Split {self.split} not in [train|test|val] for split_type subjects")
            self.subjects = subjects

        image_names = []
        sample_infos = []
        action_idxs, obj_idxs = [], []
        subject_map = {'Subject_1': 'jl', 'Subject_2': 'kl', 'Subject_3': 'sp', 'Subject_4': 'hg', 'Subject_5': 'pc'}
        seq_idx_counter = 0  # moved

        self.subjects_infos = {
            key: self.subjects_infos[key]
            for key in self.subjects_infos.keys()
            if key in self.subjects
        }

        for sub in self.subjects:
            # print(f"\n[INFO] Processing subject: {sub}")
            # print("→ self.subjects_infos keys:", list(self.subjects_infos.keys()))

            for name in self.subjects_infos[sub]:
                subject = name.split('_')[-1].lower()
                action_name = name.split('_')[0]
                frame_idx, object = self.subjects_infos[sub][name]

                # print(f"  → Action: {action_name}, Subject: {subject}, Frames: {frame_idx}, Objects: {object}")

                for iidx in range(int(frame_idx)):
                    # relative_img_path = os.path.join(
                    #     self.root , action_name, subject, name,
                    #     "RGB_undistorted", "processed_270_480",
                    #     self.rgb_template.format(iidx)
                    # )
                    relative_img_path = os.path.join(
                        "../data_MHAV", action_name, subject, name,
                        "RGB_undistorted", "processed_270_480",
                        self.rgb_template.format(iidx)
                    )
                    image_names.append(relative_img_path)
                    
                    sample_infos.append({
                        "subject": subject,
                        "action_name": action_name,
                        "frame_idx": iidx,
                        "seq_idx" : seq_idx_counter,
                        "detail_action_name": name
                    
                    })

                    action_idx = self.action_info[action_name]["action_idx"]
                    action_idxs.append(action_idx)

                    obj_idx = list(map(int, object.split(',')))
                    obj_idxs.append(obj_idx)

                seq_idx_counter += 1  # ✅ sequence 단위 증가

        annotations = {
            "image_names": image_names,
            "sample_infos": sample_infos,
            "action_idxs": action_idxs,
            "seq_idx" : seq_idx_counter,
            "video_lens": self.subjects_infos,
            "object_infos": obj_idxs
        }


        # Store to class
        self.image_names = annotations["image_names"]
        self.sample_infos = annotations["sample_infos"]
        self.action_idxs = torch.tensor(annotations["action_idxs"], dtype=torch.long)
        self.video_lens = annotations["video_lens"]
        self.obj_idxs = annotations["object_infos"]

        # Sliding window mapping
        window_starts, fulls = mhavutils_gibson.get_seq_map(
            sample_infos=self.sample_infos,
            video_lens=self.video_lens,
            ntokens_action=self.ntokens_action,
            spacing=self.spacing,
            is_shifting_window=self.is_shifting_window
        )
        self.window_starts = window_starts
        self.fulls = fulls

        # print(f"\n[DEBUG] Sliding windows: {len(window_starts)} generated")
        # print("→ Example start indices:", window_starts[:10])

    def get_start_frame_idx(self, idx):
        idx=min(idx,len(self.window_starts)-1)
        return self.window_starts[idx]
            
    
    def get_dataidx(self, idx):
        idx=min(idx,len(self.fulls)-1)
        return self.fulls[idx]

    def open_seq_lmdb(self,idx):
        return self.get_image(idx)

    def get_image(self, idx, txn=None):
        idx = self.get_dataidx(idx)
        img_path = self.image_names[idx]
        
        img_path = os.path.join(self.rgb_root, img_path)
        img = Image.open(img_path).convert("RGB")
        return img
    

    def get_hand_label(self, idx):  # hand type id 
        idx = self.get_dataidx(idx)
        img_path = self.image_names[idx]
        path_info = img_path.split('/')
        scene = path_info[2]
        subject = path_info[3]
        sequence = path_info[4]
        # import pdb;pdb.set_trace()
        frame_number = int(path_info[-1].split('.')[0].split('_')[-1])

        both_labels = []

        try:
            left_frame_labels = self.hand_labels[subject][scene][sequence][0]
            right_frame_labels = self.hand_labels[subject][scene][sequence][1]
        except KeyError:
            # print('[KeyError] img path ===>', img_path)
            left_frame_labels = ''
            right_frame_labels = ''

        # ---------------- LEFT ------------------
        for frame_range in left_frame_labels.split(','):
            if not frame_range.strip():
                continue

            range_label = frame_range.split(':')
            try:
                if '-' in range_label[0]:
                    parts = range_label[0].split('-')
                    if len(parts) != 2:
                        raise ValueError("Invalid dash format")
                    start_frame = int(parts[0])
                    last_frame = int(parts[1])
                elif len(range_label) >= 2 and all(x.strip().lstrip('-').isdigit() for x in range_label[:2]):
                    start_frame = int(range_label[0])
                    last_frame = int(range_label[1])
                else:
                    raise ValueError(f"[Invalid Label] idx={idx}, range_label={range_label}")
            except Exception as e:
                print(f"[Label Parse Error] LEFT idx={idx}, range_label={range_label} → img_path {img_path}")
                continue

            if start_frame <= frame_number <= last_frame:
                try:
                    hand_label = int(range_label[1])
                    # print("img_path", img_path, "left_hand_lebel", hand_label)
                except Exception as e:
                    print(f'[ValueError] LEFT hand_label parse error: {e} | img path: {img_path}')
                    hand_label = -1
                both_labels.append(self.mhav_hand_map.get(hand_label, 0))

        if len(both_labels) == 0:
            both_labels.append(0)

        # ---------------- RIGHT ------------------
        for frame_range in right_frame_labels.split(','):
            if not frame_range.strip():
                continue

            range_label = frame_range.split(':')
            try:
                if '-' in range_label[0]:
                    parts = range_label[0].split('-')
                    if len(parts) != 2:
                        raise ValueError("Invalid dash format")
                    start_frame = int(parts[0])
                    last_frame = int(parts[1])
                elif len(range_label) >= 2 and all(x.strip().lstrip('-').isdigit() for x in range_label[:2]):
                    start_frame = int(range_label[0])
                    last_frame = int(range_label[1])
                else:
                    raise ValueError(f"[Invalid Label] idx={idx}, range_label={range_label}")
            except Exception as e:
                print(f"[Label Parse Error] RIGHT idx={idx}, range_label={range_label} → img_path {img_path}")
                continue

            if start_frame <= frame_number <= last_frame:
                try:
                    hand_label = int(range_label[1])
                    # print("img_path", img_path, "right_hand_lebel", hand_label)

                except Exception as e:
                    print(f'[ValueError] RIGHT hand_label parse error: {e} | img path: {img_path}')
                    hand_label = -1
                both_labels.append(self.mhav_hand_map.get(hand_label, 0))

        if len(both_labels) == 1:
            both_labels.append(0)

        return both_labels


    def get_camintr(self, idx):
        idx = self.get_dataidx(idx)
        camintr = self.cam_intr
        return camintr.astype(np.float32)
 
    def get_action_idxs(self, idx):
        idx=self.get_dataidx(idx)
        action_idx=self.action_idxs[idx]
        return action_idx
    
    def get_obj_idxs(self, idx):
        idx=self.get_dataidx(idx)
        object_idx=self.obj_idxs[idx]
        return object_idx

    def get_sample_info(self,idx):
        idx=self.get_dataidx(idx)
        sample_info=self.sample_infos[idx]
        return sample_info
    
    
    def get_future_frame_idx(self, cur_idx, fut_idx, spacing, verbose=False):
        cur_idx=self.get_dataidx(cur_idx)
        fut_idx=self.get_dataidx(fut_idx)
        cur_sample_info=self.sample_infos[cur_idx]
        fut_sample_info=self.sample_infos[fut_idx]

        if int(fut_sample_info["frame_idx"])-int(cur_sample_info["frame_idx"]) != spacing:
            fut_idx=cur_idx
            not_padding=0
        else:
            not_padding=1 
        return fut_idx,not_padding


    def __len__(self):
        return len(self.window_starts)


    def __getitem__(self, idx):
        idx = self.get_dataidx(idx)

        sample = {}

        # === 기본적으로 필요한 것들 ===
        img = self.get_image(idx)
        action_idx = self.get_action_idxs(idx)
        obj_idx = self.get_obj_idxs(idx)

        # === 추가: Camera Intrinsics ===
        camintr = torch.tensor(self.cam_intr, dtype=torch.float32).clone()

        # === sample 딕셔너리에 다 넣기 ===
        sample[TransQueries.IMAGE] = img
        sample[TransQueries.ACTIONIDX] = action_idx
        sample[TransQueries.OBJIDX] = obj_idx
        sample[TransQueries.CAMINTR] = torch.tensor(self.cam_intr, dtype=torch.float32).clone()


        # 필요하면 joints2d, jointsabs25d 등도 여기에 추가 가능
    

        return sample