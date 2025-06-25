import glob
import os,cv2
import numpy as np
import fnmatch
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import lmdb
import os
import fnmatch
import cv2
import multiprocessing
from datasets import fhbhands, h2ohands, mhavhands
#Resize original fpha imgs to (480,270)
def resize_imgs_to_480_270_fpha(fhb_root="../fpha/"):

    fhb_overlay_src = os.path.join(fhb_root, "Video_files")
    fhb_overlay_dst = os.path.join(fhb_root, "Video_files_480")


    def convert(src, dst, out_size=(480, 270)):
        dst_folder = os.path.dirname(dst)
        os.makedirs(dst_folder, exist_ok=True)
        if not os.path.exists(dst):
            img = Image.open(src)
            dest_img = img.resize(out_size, Image.BILINEAR)
            dest_img.save(dst)

    subjects = [f"Subject_{subj_idx}" for subj_idx in range(1, 7)]
    # Gather all frame paths to convert
    frame_pairs = []
    for subj in subjects:
        subj_path = os.path.join(fhb_overlay_src, subj)
        actions = sorted(os.listdir(subj_path))
        for action in actions:
            action_path = os.path.join(subj_path, action)
            sequences = sorted(os.listdir(action_path))
            for seq in sequences:
                seq_path = os.path.join(action_path, seq, "color")
                frames = sorted(os.listdir(seq_path))
                for frame in frames:
                    frame_path_src = os.path.join(seq_path, frame)
                    frame_path_dst = os.path.join(fhb_overlay_dst, subj, action, seq, "color", frame)
                    frame_pairs.append((frame_path_src, frame_path_dst))

        # Resize all images
        nworkers=10
        print(f"Launching conversion for {len(frame_pairs)}")
        Parallel(n_jobs=nworkers, verbose=5)(
            delayed(convert)(frame_pair[0], frame_pair[1]) for frame_pair in frame_pairs
        )
#Resize original h2o imgs to (480,270)
def resize_imgs_to_480_270(h2o_root="../h2o"):
    list_dirs=glob.glob(os.path.join(h2o_root,'./*/*/*/cam4/'))
    for cdir in tqdm(list_dirs):
        out_dir=os.path.join(cdir,'overlay480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'overlay','*.png'))
        for im_id in range(0,len(list_imgs)):
            path_cimg=os.path.join(cdir,'overlay','{:06d}.png'.format(im_id))
            cimg=cv2.imread(path_cimg)
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'overlay480_270','{:06d}.png'.format(im_id)),cimg)

#Resize original mhav bins to (480,270)

class deserializeMatbin:
    def __init__(self, filename):
        with open(filename, 'rb') as fp:
            try:
                data_array = np.fromfile(fp, np.int32, count=4)
                self.cols = data_array[0]  # width
                self.rows = data_array[1]  # height
                self.elemSize = data_array[2]  # bytes per element (2)
                self.elemType = data_array[3]  # type indicator (0, 2, 16)

                # ✅ 채널 수 설정
                self.channel = 1  # 기본값: Grayscale
                if self.elemType == 16:  # overlay 3 채널
                    self.channel = 3

                print(f"📂 Loading: {filename}")
                print(f"📏 Original Size: {self.cols}x{self.rows}, Channels: {self.channel}, Type: {self.elemType}")

                # 데이터 타입 결정
                dtype = np.uint8 if self.elemType in [0, 16] else np.uint16

                # 이미지 데이터 불러오기
                outputMat = np.fromfile(fp, dtype)

                expected_size = self.cols * self.rows * self.channel
                actual_size = len(outputMat)

                if actual_size < expected_size:
                    print(f"❌ Error: Data size mismatch! Expected {expected_size} bytes, but got {actual_size}.")
                    return

                # 이미지 변환
                self.outputMat = outputMat[:expected_size].reshape((self.rows, self.cols, self.channel))
                print(f"✅ Successfully loaded image of shape {self.outputMat.shape}")

            except Exception as e:
                print(f"❌ Error while loading {filename}: {str(e)}")
                self.outputMat = None

    def resize_and_save(self, output_path, target_size=(480, 270)):
        """ 이미지를 270x480으로 리사이즈하고 PNG로 저장 """
        if self.outputMat is None:
            print(f"⚠ Skipping {output_path} due to load failure.")
            return

        print(f"🔄 Resizing image to {target_size}...")
        resized_img = cv2.resize(self.outputMat, target_size, interpolation=cv2.INTER_LINEAR)

        # 저장 경로 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 저장 (16bit는 8bit로 변환)
        if self.elemType == 2:
            resized_img = (resized_img / 256).astype(np.uint8)

        cv2.imwrite(output_path, resized_img)
        print(f"✅ Resized image saved at: {output_path}")



import os
import cv2
import multiprocessing
from tqdm import tqdm

def find_thermal_raw_folders(root_dir):
    """root_dir 내부에서 'warped_thermal' 폴더를 찾아 반환 (재귀 탐색)"""
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            if entry.name == "warped_thermal":     # 찾을 단어 넣기
                yield entry.path  # 'warped_thermal' 폴더 경로 반환
            else:
                yield from find_thermal_raw_folders(entry.path)  # 재귀적으로 하위 폴더 탐색

def get_jpg_files(overlay_folder):
    """warped_thermal 폴더 내 모든 .jpg 파일 경로 가져오기"""
    return [os.path.join(overlay_folder, f) for f in os.listdir(overlay_folder) if f.endswith(".jpg")]

def process_image(jpg_path):
    try:
        image = cv2.imread(jpg_path)
        if image is None or image.size == 0:
            print(f"❌ Failed to load image: {jpg_path} (File might be corrupted)")
            return

        resized_image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)

        base_dir = os.path.dirname(jpg_path)
        processed_dir = os.path.join(base_dir, "processed_270_480")
        os.makedirs(processed_dir, exist_ok=True)

        output_path = os.path.join(processed_dir, os.path.basename(jpg_path))
        cv2.imwrite(output_path, resized_image)

    except Exception as e:
        print(f"❌ Failed to process {jpg_path}: {str(e)}")

def resize_images(data_root, num_workers=None):
    print("🔍 Searching for 'warped_thermal' folders...")

    # "warped_thermal" 폴더만 찾기
    overlay_folders = list(find_thermal_raw_folders(data_root))

    if not overlay_folders:
        print("❌ No 'warped_thermal' folders found.")
        return

    print(f"📂 Found {len(overlay_folders)} 'warped_thermal' folders.")

    # 해당 폴더 내부의 .jpg 파일 수집
    jpg_files = []
    for folder in overlay_folders:
        jpg_files.extend(get_jpg_files(folder))

    if not jpg_files:
        print("❌ No .jpg files found in 'warped_thermal' folders.")
        return

    print(f"🖼️ Found {len(jpg_files)} .jpg images.")

    # 멀티프로세싱 설정
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    print(f"🛠 Using {num_workers} workers for multiprocessing.")

    # 멀티프로세싱으로 이미지 처리
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_image, jpg_files), total=len(jpg_files), desc="Processing Images"))

    print("🚀 All .jpg files have been successfully resized to 270x480 using multiprocessing!")




def resize_thermal_mask_to_480_270(data_root="../data_MHAV"):
    list_dirs = glob.glob(os.path.join(data_root, './subject*/*/'))
    tasks = []
    for cdir in list_dirs:
        out_dir = os.path.join(cdir, 'mask480_270')
        os.makedirs(out_dir,exist_ok=True)
        shutil.rmtree(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs = sorted(glob.glob(os.path.join(cdir, 'mask', '*.png')))
        for im_id, img_path in enumerate(list_imgs):
            tasks.append((img_path, os.path.join(out_dir, '{:06d}.png'.format(im_id)), (480, 270)))
    
    with ProcessPoolExecutor(24) as executor:
        executor.map(resize_image, tasks)



#Use lmdb to save image, and facilitate training
def convert_dataset_split_to_lmdb(dataset_name,dataset_folder,split):
    input_res = (480, 270)
    if dataset_name == "fhbhands":
        pose_dataset = fhbhands.FHBHands(dataset_folder=dataset_folder,
                                        split=split,
                                        ntokens_pose=16,
                                        ntokens_action=128,
                                        spacing=2,
                                        is_shifting_window=True,
                                        split_type="actions",)
                                                
    elif dataset_name=='h2ohands':
        pose_dataset = h2ohands.H2OHands(dataset_folder=dataset_folder,
                                        split=split,
                                        ntokens_pose=16,
                                        ntokens_action=128,
                                        spacing=2,
                                        is_shifting_window=True,
                                        split_type="actions",)

    # ours MHAV dataset
    elif dataset_name=='mhav':
        pose_dataset=mhavhands.MHAVhands(dataset_folder=dataset_folder,
                            split=split,
                            ntokens_action=128,
                            spacing=2,
                            is_shifting_window=True,
                            split_type="subjects")

    image_names = pose_dataset.image_names
    sample_infos=pose_dataset.sample_infos


    rgb_root = pose_dataset.rgb_root
    image_path = os.path.join(rgb_root,image_names[0])
    data_size_per_img= np.array(Image.open(image_path).convert("RGB")).nbytes 
    data_size=data_size_per_img*len(image_names)

    dir_lmdb=os.path.join(dataset_folder,'lmdb_imgs',split)
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*10,max_dbs=1000)
    pre_seq_tag=''
    commit_interval=100
    for idx in range(0,len(image_names)):
        # if dataset_name=='fhbhands':

        if dataset_name=='mhav':
            cur_seq_tag='_'.join(image_names[idx].split('/')[:-1])
        # else:
            # cur_seq_tag='{:04d}'.format(sample_infos[idx]["seq_idx"])
        if cur_seq_tag!=pre_seq_tag:
            pre_seq_tag=cur_seq_tag
            print(cur_seq_tag)

            if idx>0:
                txn.commit()
        
            subdb=env.open_db(cur_seq_tag.encode('ascii'))
            txn=env.begin(db=subdb,write=True)

        key_byte = image_names[idx].encode('ascii')
        image_path = os.path.join(rgb_root,image_names[idx])
        data = np.array(Image.open(image_path).convert("RGB"))
        # print(idx,image_names[idx])
        txn.put(key_byte,data)

        if (idx+1)%commit_interval==0:
            txn.commit()
            txn=env.begin(db=subdb,write=True)

    txn.commit()
    env.close()


if __name__ == "__main__":
    # ===== resize images to (480, 270) =====
    # resize_imgs_to_480_270_fpha(fhb_root='/data/fphab/')

    # ===== dataset split train =====
    # convert_dataset_split_to_lmdb(dataset_name='fhbhands', dataset_folder='/data/fphab/', split='train')

    # ===== dataset split test =====
    # convert_dataset_split_to_lmdb(dataset_name='fhbhands', dataset_folder='/data/fphab/', split='test')
    
    # ===== resize images to (480, 270) =====
    # resize_imgs_to_480_270(h2o_root="/data/h2o/")

    # ===== dataset split train =====
    # convert_dataset_split_to_lmdb(dataset_name='h2ohands', dataset_folder='/data/h2o/', split='train')

    # ===== dataset split test =====
    # convert_dataset_split_to_lmdb(dataset_name='h2ohands', dataset_folder='/data/h2o/', split='val')


    # ===== resize images to (480, 270) =====
    resize_images(data_root="../data_MHAV", num_workers=12)


    # ===== dataset split train =====
    # convert_dataset_split_to_lmdb(dataset_name='mhav', dataset_folder="../data_MHAV", split='train')

    # ===== dataset split test =====
    # convert_dataset_split_to_lmdb(dataset_name='mhav', dataset_folder="../data_MHAV", split='test')