import os
import glob
import shutil

def organize_bin_files(base_dir):
    """
    🔹 `base_dir` 내부의 모든 `.bin` 파일을 찾아 유형별로 정리하는 함수
    🔹 각 `.bin` 파일이 존재하는 폴더 내부에 `depth/`, `rgb/`, `thermal/` 폴더를 생성하고 파일 이동
    """
    # `base_dir` 아래의 모든 하위 폴더 검색
    all_dirs = [d for d in glob.glob(os.path.join(base_dir, "*", "*", "*")) if os.path.isdir(d)]

    for target_dir in all_dirs:
        print(f" Processing: {target_dir}")

        # 정리할 폴더 경로 생성
        depth_dir = os.path.join(target_dir, "depth")
        rgb_dir = os.path.join(target_dir, "rgb")
        thermal_dir = os.path.join(target_dir, "thermal")

        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(thermal_dir, exist_ok=True)

        # 파일 이동 함수
        def move_files(pattern, destination):
            files = glob.glob(os.path.join(target_dir, pattern))
            for file in files:
                shutil.move(file, os.path.join(destination, os.path.basename(file)))
                print(f"✅ Moved: {file} → {destination}")

        # 파일 정리
        move_files("rawDepth_*.bin", depth_dir)
        move_files("RGB_*.bin", rgb_dir)
        move_files("thermal_*.bin", thermal_dir)

    print("🚀 All `.bin` files have been organized into separate folders!")

# 실행
# base_dir = "/mnt/DI_hdd/BMVC/database_merge"
organize_bin_files(base_dir)
