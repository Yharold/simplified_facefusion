import os
import time
from back_end.face_analyser import get_many_faces
from back_end.modules.face_swapper import process_image, process_video
from mid_end.face_store import append_reference_faces
from mid_end.ffmpeg import extract_frames
from mid_end.vision import get_temp_dir, read_image
from mid_end import process_manager


def test_swapper():
    source_image_paths = [r"input\glt1.png", r"input\glt2.png", r"input\glt3.png"]
    target_image_path = r"input\zjl4.jpg"
    image = read_image(target_image_path)

    faces = get_many_faces(image)
    for face in faces:
        append_reference_faces("test", face)
    process_image(source_image_paths, target_image_path, "output/temp1.jpg")


def test_video_swapper():
    video_path = r"input\target-240p.mp4"
    extract_frames(video_path)
    time.sleep(1.0)
    temp_path = get_temp_dir(video_path)
    temp_frame_paths = [
        os.path.join(temp_path, item)
        for item in os.listdir(temp_path)
        if item.endswith((".png"))
    ]
    image_path = temp_frame_paths[0]
    image = read_image(image_path)
    faces = get_many_faces(image)
    for face in faces:
        append_reference_faces("test", face)
    process_manager.set_process_state("processing")
    source_paths = [r"input\glt1.png", r"input\glt2.png", r"input\glt3.png"]
    process_video(source_paths, temp_frame_paths)
 
test_swapper()
