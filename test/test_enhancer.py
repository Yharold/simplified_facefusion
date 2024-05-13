import os
from back_end.face_analyser import get_many_faces
from back_end.modules.face_enhancer import process_image, process_video
from mid_end.face_store import append_reference_faces
from mid_end.vision import read_image
from mid_end import process_manager


def test_enhancer():
    target_image_path = r"temp\target-240p\0006.png"
    target_image = read_image(target_image_path)
    faces = get_many_faces(target_image)
    for face in faces:
        append_reference_faces("test", face)
    process_image(None, target_image_path, "output/0006.png")


def test_video_enhancer():
    temp_dir = r"temp\target-240p"
    temp_frame_paths = [os.path.join(temp_dir, item) for item in os.listdir(temp_dir)]
    image_path = temp_frame_paths[0]
    image = read_image(image_path)
    faces = get_many_faces(image)
    for face in faces:
        append_reference_faces("test", face)
    process_manager.set_process_state("processing")
    process_video(None, temp_frame_paths)


test_enhancer()
