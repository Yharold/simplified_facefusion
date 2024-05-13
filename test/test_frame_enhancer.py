import os
from back_end.face_analyser import get_many_faces
from back_end.modules.frame_enhancer import process_image, process_video
from mid_end.face_store import append_reference_faces
from mid_end.vision import read_image
from mid_end import process_manager


def test_image():
    target_image_path = r"input\8.jpg"
    process_image(None, target_image_path, "output/8-3.png")


def test_video():
    temp_dir = r"temp\target-240p_1"
    temp_frame_paths = [os.path.join(temp_dir, item) for item in os.listdir(temp_dir)]
    process_manager.set_process_state("processing")
    process_video(None, temp_frame_paths[0:10])


test_image()
