import os
from mid_end.vision import read_image
from back_end.modules.frame_colorizer import process_image, process_video
from mid_end import process_manager


def test_image():
    target_image_path = r"input\4.jpg"
    process_image(None, target_image_path, "output/4-2.png")


def test_video():
    temp_dir = r"temp\target-240p_1"
    temp_frame_paths = [os.path.join(temp_dir, item) for item in os.listdir(temp_dir)]
    process_manager.set_process_state("processing")
    process_video(None, temp_frame_paths[12:22])


test_image()
