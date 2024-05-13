import os
import cv2
from mid_end.ffmpeg import extract_frames, merge_video
from mid_end import process_manager


def test_ffmpeg():
    video_path = r"input\v1.mp4"
    process_manager.set_process_state("processing")
    code = extract_frames(video_path)
    print(code)


def test_merge():
    target_path = r"temp\target-240p"
    image_path = os.path.join(target_path, os.listdir(target_path)[0])
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    resolution = str(width) + "x" + str(height)
    fps = 25.0
    process_manager.set_process_state("processing")
    a = merge_video(target_path, resolution, fps)
    print(a)

test_merge()
