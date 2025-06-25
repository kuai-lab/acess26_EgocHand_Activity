from collections import defaultdict
import os

import numpy as np
from tqdm import tqdm  
import pandas as pd

def get_seq_map(sample_infos, video_lens, ntokens_action, spacing, is_shifting_window): 
    subject_map = {'jl': 'Subject_1', 'kl': 'Subject_2', 'sp': 'Subject_3', 'hg': 'Subject_4', 'pc': 'Subject_5'}
    
    cur_sample = sample_infos[0]
    pre_key = f"{cur_sample['action_name']}_hand_{cur_sample['subject']}"
    
    seq_count = 0
    cur_seq_len = video_lens.get(pre_key, 0)  # 키가 없을 수도 있음

    window_starts = []
    full = []
    
    for sample_idx, sample_info in enumerate(sample_infos):
        cur_key = f"{sample_info['action_name']}_hand_{sample_info['subject']}"

        if pre_key != cur_key:
            pre_key = cur_key
            seq_count = 0
            cur_seq_len = video_lens.get(pre_key, 0)  
                
        if is_shifting_window and seq_count % (ntokens_action * spacing) < spacing:
            window_starts.append(sample_idx)        
            
        full.append(sample_idx)
        seq_count += 1
    
    return window_starts, full



def get_action_train_test(action_info, subjects_info):
    all_infos ={}
    for subject in subjects_info:
        for item in subjects_info[subject]:
            action_name = item[0]
            seq_num = int(item[1])
            frame_nb = subjects_info[subject][item]
            for frame_idx in range(int(frame_nb)):
                all_infos[(subject, action_name, seq_num, frame_idx)] =action_info[action_name]['action_idx']
    return all_infos


def get_action_infos(path_action_info):
    action_info = {}
    tmp={}
    with open(path_action_info, "r") as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        action_id, action_name = line.strip().split()
            
        if action_name not in action_info:
            action_info[action_name]={}
            action_info[action_name]['action_idx'] =int(action_id)
            tmp[action_name]=int(action_id)
    
    return action_info, tmp


def get_object_infos(path_object_info):
    object_info = {}
    tmp={}
    with open(path_object_info, "r") as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        object_id, object_name = line.strip().split()
            
        if object_name not in object_info:
            object_info[object_name]={}
            object_info[object_name]['object_idx'] =int(object_id)
            tmp[object_name]=int(object_id)
    
    return object_info, tmp

def get_hand_type_info(path_hand_type_info):
    hand_type_info={}
    with open(path_hand_type_info, "r") as f:
        raw_lines = f.readlines()
        for line in raw_lines[1:]:
            line = " ".join(line.split())
            type_idx, type_name = line.strip().split(" ")
            hand_type_info[type_name] = {'type_idx': int(type_idx)}

    return hand_type_info


# multi-modal(depth, thermal) 용도로 변경
import os
import pandas as pd

def get_all_hand_labels(file_path, rgb_root, rgb_template, depth_root, depth_template, thermal_root, thermal_template): 
    info = dict()
    subjects_infos = {}
    df = pd.read_excel(file_path)

    for idx, (name, l_info, r_info, object) in enumerate(zip(df['DATA_MHAV'], df.L, df.R, df.object)):
        name = name.strip()
        parts = name.split("_")
        action = parts[0]
        subject = parts[-1].lower()

        save = True
        reason = ""

        # 👈 왼쪽 라벨 유효성 검사
        if isinstance(l_info, str):
            for item in l_info.split(','):
                tmp = item.split(':')[-1].strip()
                if ';' in tmp:
                    tmp = tmp.split(';')[-1].strip()
                try:
                    tmp = int(tmp)
                    if not (-2 < tmp < 100):
                        save = False
                        reason = f"[LEFT LABEL RANGE] {tmp} out of range"
                except:
                    save = False
                    reason = "[LEFT LABEL PARSE ERROR]"
        else:
            save = False
            reason = "[LEFT LABEL NOT STR]"

        # 👈 오른쪽 라벨 유효성 검사
        if isinstance(r_info, str):
            for item in r_info.split(','):
                temp = item.split(':')[-1].strip()
                if ';' in temp:
                    temp = temp.split(';')[-1].strip()
                try:
                    temp = int(temp)
                    if not (-2 < temp < 100):
                        save = False
                        reason = f"[RIGHT LABEL RANGE] {temp} out of range"
                except:
                    save = False
                    reason = "[RIGHT LABEL PARSE ERROR]"
        else:
            save = False
            reason = "[RIGHT LABEL NOT STR]"

        # 👈 프레임 폴더 및 멀티모달 이미지 존재 여부 체크
        if save:
            base_dir = os.path.join(action, subject, name)
            rgb_dir = os.path.join(rgb_root, base_dir, "RGB_undistorted", "processed_270_480")
            depth_dir = os.path.join(depth_root, base_dir, "Depth_new", "processed_270_480")
            thermal_dir = os.path.join(thermal_root, base_dir, "warped_thermal", "processed_270_480")

            if os.path.exists(rgb_dir):
                frames = sorted(os.listdir(rgb_dir))
                last_frame_idx = len(frames) - 1

                try:
                    l_max = int(l_info.split(':')[-2].split('-')[-1])
                    r_max = int(r_info.split(':')[-2].split('-')[-1])
                except:
                    save = False
                    reason = "[FRAME INDEX PARSE ERROR]"
                    continue

                if abs(last_frame_idx - l_max) <= 1 and abs(last_frame_idx - r_max) <= 1:
                    for frame_idx in range(len(frames)):
                        rgb_path = os.path.join(rgb_dir, rgb_template.format(frame_idx))
                        depth_path = os.path.join(depth_dir, depth_template.format(frame_idx))
                        thermal_path = os.path.join(thermal_dir, thermal_template.format(frame_idx))

                        if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(thermal_path)):
                            save = False
                            reason = f"[MISSING FILE] RGB or Depth or Thermal missing at frame {frame_idx}"
                            break
                else:
                    save = False
                    reason = f"[FRAME COUNT MISMATCH] last={last_frame_idx}, l_max={l_max}, r_max={r_max}"
            else:
                save = False
                reason = f"[MISSING FOLDER] {rgb_dir}"

        # 👈 저장 or 스킵 처리
        if not save:
            print(f"[SKIPPED:{idx}] {name} - {reason}")
            continue

        # ✅ 저장
        if subject not in info:
            info[subject] = {}
        if action not in info[subject]:
            info[subject][action] = {}
        info[subject][action][name] = (l_info, r_info)

        if subject not in subjects_infos:
            subjects_infos[subject] = {}
        subjects_infos[subject][name] = (str(len(frames)), object)

    print("[Final] subjects_infos keys:", list(subjects_infos.keys()))
    return info, subjects_infos
