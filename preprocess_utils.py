import glob
import os,cv2
import numpy as np
import fnmatch
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import lmdb
from datasets import fhbhands, h2ohands, mhavhands
#Resize original fpha imgs to (480,270)
def resize_imgs_to_480_270_fpha(fhb_root="../fpha/"):

    fhb_rgb_src = os.path.join(fhb_root, "Video_files")
    fhb_rgb_dst = os.path.join(fhb_root, "Video_files_480")


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
        subj_path = os.path.join(fhb_rgb_src, subj)
        actions = sorted(os.listdir(subj_path))
        for action in actions:
            action_path = os.path.join(subj_path, action)
            sequences = sorted(os.listdir(action_path))
            for seq in sequences:
                seq_path = os.path.join(action_path, seq, "color")
                frames = sorted(os.listdir(seq_path))
                for frame in frames:
                    frame_path_src = os.path.join(seq_path, frame)
                    frame_path_dst = os.path.join(fhb_rgb_dst, subj, action, seq, "color", frame)
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
        out_dir=os.path.join(cdir,'rgb480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'rgb','*.png'))
        for im_id in range(0,len(list_imgs)):
            path_cimg=os.path.join(cdir,'rgb','{:06d}.png'.format(im_id))
            cimg=cv2.imread(path_cimg)
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'rgb480_270','{:06d}.png'.format(im_id)),cimg)

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
                if self.elemType == 16:  # RGB 3 채널
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



def resize_bin_images(data_root):
    """
    🔹 `data_root` 내의 모든 .bin 파일을 찾아 PNG로 변환 후 저장하는 함수
    🔹 변환 후 `.bin`이 있는 폴더 내부에 `processed_270_480/` 폴더를 만들고 PNG 저장
    """
    # # 모든 .bin 파일 찾기
    # bin_files = glob.glob(os.path.join(data_root, "**", "RGB_*.bin"), recursive=True)
    # print(len(bin_files))
    # 먼저 10개만 출력
    
    print("🔍 Searching for .bin files...")

    bin_files = []
    
    for root, _, files in os.walk(data_root):
        rgb_bins = fnmatch.filter(files, "RGB_*.bin")       # ㅈㄴ 오래걸려서 계속 시도중
        for bin_file in rgb_bins:
            bin_files.append(os.path.join(root, bin_file))


    for idx, bin_path in enumerate(bin_files):
        try:
            print(f"🔍 [{idx+1}/{len(bin_files)}] Processing: {bin_path}")

            # `.bin` 파일을 PNG로 변환
            image = deserializeMatbin(bin_path)

            # `.bin`이 있는 폴더 내부에 `processed_270_480` 폴더 생성
            base_dir = os.path.dirname(bin_path)  # `.bin` 파일이 위치한 폴더
            processed_dir = os.path.join(base_dir, "processed_270_480")  # 변환된 이미지 저장 폴더

            os.makedirs(processed_dir, exist_ok=True)  # 폴더가 없으면 생성

            png_filename = os.path.basename(bin_path).replace(".bin", ".png")
            output_path = os.path.join(processed_dir, png_filename)

            # 변환 후 저장
            image.resize_and_save(output_path, target_size=(480, 270))

        except Exception as e:
            print(f"❌ Failed to process {bin_path}: {str(e)}")

    print("🚀 All .bin files have been successfully converted to 270x480 PNG images!")



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
        pose_dataset=mhavhands.MHAVhands(dataset_folder="../data_MHAV/",
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
        if dataset_name=='fhbhands':
            cur_seq_tag='_'.join(image_names[idx].split('/')[:-1])
        else:
            cur_seq_tag='{:04d}'.format(sample_infos[idx]["seq_idx"])
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
        print(idx,image_names[idx])
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
    # resize_bin_images(data_root="../database_merge/")


    # ===== dataset split train =====
    convert_dataset_split_to_lmdb(dataset_name='mhav', dataset_folder="../data_MHAV/", split='train')

    # ===== dataset split test =====
    # convert_dataset_split_to_lmdb(dataset_name='mhav', dataset_folder"../data_MHAV/", split='test')