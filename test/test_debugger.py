import os
import time
from back_end.face_analyser import find_similar_faces, get_many_faces
from back_end.modules.face_debugger import debug_face, process_image, process_video
from matplotlib.patches import draw_bbox
import cv2
from mid_end.face_store import append_reference_faces, get_reference_faces
from mid_end.ffmpeg import extract_frames
from mid_end.vision import get_temp_dir, read_image, read_static_image
from mid_end import process_manager


def test_process_image():
    image_path = r"input\2.jpg"
    # image = read_image(image_path)
    # faces = get_many_faces(image)
    # for face in faces:
    #     append_reference_faces("test", face)
    process_image(None, image_path, "output/22.jpg")


def test_process_video():
    video_path = r"input\target-240p.mp4"
    extract_frames(video_path)
    time.sleep(1.0)
    temp_path = get_temp_dir(video_path)
    temp_frame_paths = [
        os.path.join(temp_path, item)
        for item in os.listdir(temp_path)
        if item.endswith((".png"))
    ]
    # image_path = temp_frame_paths[0]
    # image = read_image(image_path)
    # faces = get_many_faces(image)
    # for face in faces:
    #     append_reference_faces("test", face)
    process_manager.set_process_state("processing")
    process_video(None, temp_frame_paths[2:7])


def test_reference_face():
    source_path = r"input\glt1.png"
    target_path = r"input\gtl_d1.png"
    source_image = read_static_image(source_path)
    target_image = read_static_image(target_path)
    faces = get_many_faces(source_image)
    if faces:
        for face in faces:
            append_reference_faces("test", face)
    reference_face = get_reference_faces()
    similar_faces = find_similar_faces(reference_face, target_image)
    images = []
    for face in similar_faces:
        image = debug_face(face, target_image)
        images.append(image)
    for image in images:
        cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_process_video()
