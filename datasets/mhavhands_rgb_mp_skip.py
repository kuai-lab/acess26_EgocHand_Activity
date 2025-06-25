from os import path
import os
import sys
import torch
sys.path.append(path.abspath('.'))

import lmdb

import numpy as np
from PIL import Image, ImageFile
from datasets import mhavutils_rgb_mp
from datasets.queries import BaseQueries,TransQueries, get_trans_queries
from datasets import mhavutils_rgb_mp
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MHAVhands(object):
    def __init__(
        self,
        dataset_folder,
        split,#
        ntokens_action,
        spacing,
        is_shifting_window,
        split_type="actions",
    ):
        super().__init__()


        self.ntokens_action=ntokens_action
        self.spacing=spacing
        self.is_shifting_window=is_shifting_window
    

        self.all_queries = [
            BaseQueries.IMAGE,           
            BaseQueries.CAMINTR,

            TransQueries.JOINTS2D, 
            TransQueries.JOINTSABS25D,

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

        self.rgb_template = "RGB_undistorted_{}.jpg"
        self.mp_template = "keypoint_{}.csv"
        # self.thermal_template = "keypoint_{}.csv"
        print("앙 확인띠", self.root)
        small_rgb = self.root  # 또는 실제 리사이즈 이미지 경로
        small_depth = self.root
        small_thermal = self.root

        if os.path.exists(small_rgb) and self.reduce_res:
            self.rgb_root = small_rgb
            self.depth_root = small_depth
            self.thermal_root = small_thermal

            self.reduce_factor = 1 / 4
            print("✅ Using reduced resolution.")
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")  # 🔧 수정
            self.reduce_factor = 1
            print("⚠️ Warning: reduce factor is 1. Using full-res images.")


        #Load action labels
        path_action_info = '../data_MHAV/action_info.txt'
        action_info, action_to_idx = mhavutils_rgb_mp.get_action_infos(path_action_info)
        self.action_info=action_info
        self.action_to_idx=action_to_idx
        self.num_actions = len(self.action_info.keys())

        #Load object labels
        path_object_info = '../data_MHAV/object_tool_id.txt'
        object_info, object_to_idx = mhavutils_rgb_mp.get_object_infos(path_object_info)
        self.object_info=object_info
        self.object_to_idx=object_to_idx
        self.num_objects = len(self.object_info.keys())


        # Load hand type labels
        self.num_handtypes = 39
        self.hand_labels, self.subjects_infos = mhavutils_rgb_mp.get_all_hand_labels('../data_MHAV/annotation_0317.xlsx', self.rgb_root, self.rgb_template, self.thermal_root, self.mp_template)
        self.mhav_hand_map={-1: 0,  1: 1,  2: 2,  3: 3,  4: 4,  8: 5,  9: 6,  12: 7,  14: 8,  15: 9,  17: 10,  18: 11,  22: 12,  23: 13,  27: 14, 29: 15,  32: 16,  33: 17,  34: 18,  36: 19,  38: 20,  39: 21,  40: 22,  41: 23,  42: 24,  43: 25,  44: 26,  45: 27,  46: 28,  47: 29,  48: 30,  49: 31,  50: 32,  51: 33,  52: 34,  53: 35,  54: 36,  55: 37,  56: 38}

        
        for i,(k,v) in enumerate(self.action_info.items()):
            self.action_info[k]["action_idx"]=i

        # Load skip list
        skip_path = '../data_MHAV/bidirection_bad_seqs.xlsx'
        self.skip_seqs = set()
        if os.path.exists(skip_path):
            df = pd.read_excel(skip_path, header=None, engine="openpyxl")
            skip_values = df.iloc[1:, 0].dropna().astype(str).tolist()  # 2행부터, 첫 번째 열
            self.skip_seqs = set(skip_values)
            print(f"🛑[INFO] Loaded {len(self.skip_seqs)} sequences to skip")
        else:
            print(f"[WARN] bidirection_bad_seqs.xlsx 없음, skip 안 함")

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
        # intrinsic은 중요할지도?
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

        modal_paths = []  # 멀티모달 경로를 담을 리스트로 변경
        mask_names = []
        joints2d = []
        joints3d = []

        # image_names = []
        sample_infos = []
        action_idxs, obj_idxs = [], []
        subject_map = {'Subject_1': 'jl', 'Subject_2': 'kl', 'Subject_3': 'sp', 'Subject_4': 'hg', 'Subject_5': 'pc'}
        self.subjects_infos = {
            key: self.subjects_infos[key]
            for key in self.subjects_infos.keys()
            if key in self.subjects
        }

        seq_idx_counter = 0  # moved outside
        for sub in self.subjects:
            print(f"\n[INFO] Processing subject: {sub}")
            print("→ self.subjects_infos keys:", list(self.subjects_infos.keys()))

            for name in self.subjects_infos[sub]:
                if name in self.skip_seqs:
                    # __init__ 안에 있는 skip list 로딩 후
                    print("[SKIP SEQ 목록 미리보기]")
                    for seq in sorted(list(self.skip_seqs))[:10]:
                        print(" -", seq)
                    print(f"총 {len(self.skip_seqs)}개 시퀀스가 스킵 예정입니다.")

                    print(f"[SKIP] Skipping sequence {name}")
                    continue
                subject = name.split('_')[-1].lower()
                action_name = name.split('_')[0]
                frame_idx, object = self.subjects_infos[sub][name]

                print(f"  → Action: {action_name}, Subject: {subject}, Frames: {frame_idx}, Objects: {object}")

                # for iidx in range(int(frame_idx)):
                #     base_dir = os.path.join("../data_MHAV", action_name, subject, name)

                #     modal_paths.append({
                #         "rgb": os.path.join(base_dir, "RGB_undistorted", "processed_270_480", self.rgb_template.format(iidx)),
                #         "mp": os.path.join(base_dir, "pose_2d_csv", self.mp_template.format(iidx)),
                #     })
                for iidx in range(int(frame_idx)):
                    base_dir = os.path.join("../data_MHAV", action_name, subject, name)

                    rgb_path = os.path.join(base_dir, "RGB_undistorted", "processed_270_480", self.rgb_template.format(iidx))
                    mp_path = os.path.join(base_dir, "pose_2d_mp_csv", self.mp_template.format(iidx))

                    # CSV가 없을 경우, 0으로 패딩된 좌표를 생성
                    if not os.path.exists(rgb_path):
                        continue  # RGB 이미지 자체가 없으면 스킵

                    if os.path.exists(mp_path):
                        final_mp_path = mp_path
                    else:
                        # 빈 CSV 파일 생성 (42 x 3 모두 0)
                        os.makedirs(os.path.dirname(mp_path), exist_ok=True)
                        with open(mp_path, 'w') as f:
                            f.write('x,y,z\n')
                            for _ in range(42):
                                f.write('0,0,0\n')
                        final_mp_path = mp_path

                    modal_paths.append({
                        "rgb": rgb_path,
                        "mp": final_mp_path,
                    })
                    sample_infos.append({
                        "subject": subject,
                        "action_name": action_name,
                        "frame_idx": iidx,
                        "seq_idx" : seq_idx_counter,
                    })
                    action_idx = self.action_info[action_name]["action_idx"]
                    action_idxs.append(action_idx)

                    obj_idx = list(map(int, object.split(',')))
                    obj_idxs.append(obj_idx)

                seq_idx_counter += 1

        annotations = {
            "modal_paths" : modal_paths,        # for Multi-Modal
            # "image_names": image_names,
            "sample_infos": sample_infos,
            "action_idxs": action_idxs,
            "seq_idx" : seq_idx_counter,
            "video_lens": self.subjects_infos,
            "object_infos": obj_idxs
        }

        print("\n[DEBUG] Final annotation info:")
        # print(f"→ Total images: {len(image_names)}")
        print(f"→ Total sample_infos: {len(sample_infos)}")
        print(f"→ Total action_idxs: {len(action_idxs)}")
        print(f"→ Total obj_idxs: {len(obj_idxs)}")

        # Store to class
        self.modal_paths = annotations["modal_paths"]  # ✅ image_names → modal_paths
        # self.image_names = annotations["image_names"]
        self.sample_infos = annotations["sample_infos"]
        self.action_idxs = torch.tensor(annotations["action_idxs"], dtype=torch.long)
        self.video_lens = annotations["video_lens"]
        self.obj_idxs = annotations["object_infos"]

        # Sliding window mapping
        window_starts, fulls = mhavutils_rgb_mp.get_seq_map(
            sample_infos=self.sample_infos,
            video_lens=self.video_lens,
            ntokens_action=self.ntokens_action,
            spacing=self.spacing,
            is_shifting_window=self.is_shifting_window
        )
        self.window_starts = window_starts
        self.fulls = fulls

        print(f"\n[DEBUG] Sliding windows: {len(window_starts)} generated")
        print("→ Example start indices:", window_starts[:10])

    def get_start_frame_idx(self, idx):
        idx=min(idx,len(self.window_starts)-1)
        return self.window_starts[idx]
            
    
    def get_dataidx(self, idx):
        idx=min(idx,len(self.fulls)-1)
        return self.fulls[idx]

    # def open_seq_lmdb(self,idx):
    #     return self.get_image(idx)

    def open_seq_lmdb(self,idx):
        return self.get_modal_images(idx)

    
    # def get_image(self, idx, txn=None):
    #     idx = self.get_dataidx(idx)
    #     img_path = self.image_names[idx]
        
    #     img_path = os.path.join(self.rgb_root, img_path)
    #     img = Image.open(img_path).convert("RGB")
    #     return img


    # multi modal(+depth, thermal)
    def get_modal_images(self, idx, txn = None):
        idx = self.get_dataidx(idx)
        paths = self.modal_paths[idx]

        rgb = Image.open(os.path.join(self.rgb_root, paths["rgb"])).convert("RGB")
        # depth = Image.open(os.path.join(self.depth_root, paths["depth"])).convert("RGB")
        mp = os.path.join(self.thermal_root, paths["mp"])
        # print(f"[DEBUG] Thermal path: {thermal_path}")
        return rgb, mp

    def get_mask(self, idx, txn):
        idx = self.get_dataidx(idx)
        img_path = self.mask_names[idx]
        if self.split=='train':
            buf=txn.get(img_path.encode('ascii'))
            img_flat=np.frombuffer(buf,dtype=np.uint8)
            img = img_flat.reshape(self.image_size[1],self.image_size[0],3).copy()
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
 
        else:
            img_path = os.path.join(self.root, img_path)
            img = Image.open(img_path).convert("RGB")
        return img 

    def get_hand_label(self, idx):  # hand type id 
        idx = self.get_dataidx(idx)
        img_path = self.modal_paths[idx]["rgb"]
        path_info = img_path.split('/')
        scene = path_info[4]
        subject = path_info[5]
        sequence = path_info[6]
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
