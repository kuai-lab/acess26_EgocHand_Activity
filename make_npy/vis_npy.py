import numpy as np
import matplotlib.pyplot as plt
import os

# --- ⚙️ 시각화할 파일 설정 ⚙️ ---
npy_file_path = ".../data_MHAV/assemble/hg/assemble_hexNut-metalBlock3-spacer-hexBolt_hand_hg/assemble_hexNut-metalBlock3-spacer-hexBolt_hand_hg.npy"
# 저장할 이미지 파일 이름
save_path = "./rotation_plot.png"
# --------------------------------

try:
    # 1. .npy 파일 불러오기
    rotation_data = np.load(npy_file_path)

    # 2. 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(rotation_data, label='Cumulative Rotation')

    # 3. 그래프 꾸미기
    plt.title(f"wrist Feature Visualization\n({os.path.basename(npy_file_path)})")
    plt.xlabel("Frame Index")
    plt.ylabel("Cumulative Rotation Angle (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 4. 이미지로 저장
    plt.savefig(save_path)
    print(f"✅ 그래프가 이미지로 저장되었습니다: {save_path}")

except FileNotFoundError:
    print(f"❌ Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {npy_file_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
