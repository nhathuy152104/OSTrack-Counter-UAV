import cv2
import os
import glob

root = "/home/arsene_lupin/HuyWorkspace/VtWork/dataset/Test"

# Tìm tất cả sequence có ground_truth.txt
anno_files = sorted(
    glob.glob(
        os.path.join(root, '*/groundtruth_rect.txt')
    )
)

print("Found sequences:", len(anno_files))

for anno_file in anno_files:

    # /Train/seq001/ground_truth.txt
    seq_path = os.path.dirname(anno_file)

    # Tìm file mp4 trong sequence
    video_files = glob.glob(
        os.path.join(seq_path, '*.mp4')
    )

    if len(video_files) == 0:
        print("No video:", seq_path)
        continue

    video_path = video_files[0]

    print("\nProcessing:", video_path)

    os.remove(video_path)
    