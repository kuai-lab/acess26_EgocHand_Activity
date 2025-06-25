from collections import defaultdict
import os

import numpy as np
from tqdm import tqdm  
import pandas as pd
 

# def get_seq_map(sample_infos, video_lens, ntokens_action, spacing, is_shifting_window): 
#     subject_map ={'jl': 'Subject_1', 'kl':'Subject_2', 'sp':'Subject_3', 'hg':'Subject_4', 'pc':'Subject_5'}
#     cur_sample = sample_infos[0]
#     pre_key = (subject_map[cur_sample["subject"]], cur_sample["action_name"], str(cur_sample["seq_idx"]))
#     seq_count = 0
#     cur_seq_len=video_lens[pre_key] 

#     window_starts = []
#     full = []
    
#     for sample_idx, sample_info in enumerate(sample_infos):
#         cur_key = (subject_map[sample_info["subject"]], sample_info["action_name"], str(sample_info["seq_idx"]))
#         if pre_key != cur_key:
#             pre_key=cur_key
#             seq_count=0
#             cur_seq_len=video_lens[pre_key] 
                
#         if ((is_shifting_window and seq_count%(ntokens_action*spacing)<spacing)):
#             window_starts.append(sample_idx)        
            
#         full.append(sample_idx)
#         seq_count += 1
    
#     return window_starts, full


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


# def get_all_hand_labels(file_path, rgb_root, rgb_template): 
#     info = dict()
#     subjects_infos = {}
#     df = pd.read_excel(file_path)
#     sub = []

#     for name, l_info, r_info, object in zip(df['DATA_MHAV'], df.L, df.R, df.object):
#         action, subject = name.strip().split("_")[0], name.strip().split("_")[-1]

#         if subject not in info:
#             info[subject] = {}
#         if action not in info[subject]:
#             info[subject][action] = {}

#         save = True
#         if isinstance(l_info, str):  # 문자열인 경우만 처리
#             for item in l_info.split(','):
#                 tmp = item.split(':')[-1].strip()
#                 if ';' in tmp:
#                     tmp = tmp.split(';')[-1].strip()

#                 try:
#                     tmp = int(tmp)
#                     if not (-2 < tmp < 100):
#                         save = False
#                 except:
#                     save = False
#         else:
#             save = False

#         if isinstance(r_info, str):
#             for item in r_info.split(','):
#                 temp = item.split(':')[-1].strip()
#                 if ';' in temp:
#                     temp = temp.split(';')[-1].strip()

#                 try:
#                     temp = int(temp)
#                     if not (-2 < temp < 100):
#                         save = False
#                 except:
#                     save = False
#         else:
#             save = False

#         if save:
#             data_path = os.path.join("/mnt/ssd2tb/data_MHAV", action, subject, name, "RGB_undistorted")
#             if os.path.exists(data_path):
#                 frames = os.listdir(data_path)
#                 if (len(frames)-1, len(frames)-1) == (int(l_info.split(':')[-2].split('-')[-1]), int(r_info.split(':')[-2].split('-')[-1])):
#                     for frame_idx in range(len(frames)):
#                         relative_img_path = os.path.join(action, subject, name, "RGB_undistorted", "processed_270_480", rgb_template.format(frame_idx))
#                         if not os.path.exists(os.path.join(rgb_root, relative_img_path)):
#                             save = False
#                 else:
#                     save = False
#             else:
#                 save = False

#         if save:
#             info[subject][action][name] = (l_info, r_info)
#             if subject not in sub:
#                 sub.append(subject)
#                 subjects_infos[subject] = {}

#             subjects_infos[subject][name] = (str(frame_idx), object)


#     return info, subjects_infos

def get_all_hand_labels(file_path, rgb_root, rgb_template): 
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

        # 👈 프레임 폴더 및 이미지 존재 여부 체크
        if save:
            data_path = os.path.join("../data_MHAV", action, subject, name, "RGB_undistorted", "processed_270_480")
            if os.path.exists(data_path):
                frames = sorted(os.listdir(data_path))
                last_frame_idx = len(frames) - 1

                try:
                    l_max = int(l_info.split(':')[-2].split('-')[-1])
                    r_max = int(r_info.split(':')[-2].split('-')[-1])
                except:
                    save = False
                    reason = "[FRAME INDEX PARSE ERROR]"
                    continue

                # ✅ ±1 프레임 차이까지 허용
                if abs(last_frame_idx - l_max) <= 1 and abs(last_frame_idx - r_max) <= 1:
                    for frame_idx in range(len(frames)):
                        relative_img_path = os.path.join(
                            "../data_MHAV", action, subject, name, "RGB_undistorted", "processed_270_480", rgb_template.format(frame_idx)
                        )
                        if not os.path.exists(os.path.join(rgb_root, relative_img_path)):
                            save = False
                            reason = f"[MISSING IMAGE FILE] {relative_img_path}"
                            break
                else:
                    save = False
                    reason = f"[FRAME COUNT MISMATCH] last={last_frame_idx}, l_max={l_max}, r_max={r_max}"
            else:
                save = False
                reason = f"[MISSING FOLDER] {data_path}"

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
    # print("[Final] object_infos keys:", list(_infos.keys()))
    return info, subjects_infos


def load_single_keypoint_csv(csv_path):
    import csv
    import numpy as np

    keypoints = np.zeros((42, 2), dtype=np.float32)

    if not os.path.exists(csv_path):
        # print(f"[WARN] CSV 파일 {csv_path} 없음. 0으로 채웁니다.")
        return keypoints

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header (x, y, z)

        rows = list(reader)
        num_points = len(rows)

        if num_points not in [21, 42]:
            # print(f"[WARN] keypoint 개수 한 손: {num_points}개 in {csv_path}")
            num_points = min(num_points, 42)

        for i in range(num_points):
            try:
                x, y = float(rows[i][0]), float(rows[i][1])
                keypoints[i] = [x, y]
            except Exception as e:
                print(f"[ERROR] {csv_path} 라인 {i} 파싱 실패: {e}")

    return keypoints
