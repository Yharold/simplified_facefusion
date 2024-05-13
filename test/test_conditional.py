import os
import cv2
from mid_end.core import conditional_append_reference_faces, conditional_process
from mid_end import globals, process_manager
from mid_end.ffmpeg import extract_frames, merge_video, restore_audio
from mid_end.vision import (
    clear_temp,
    detect_video_fps,
    detect_video_resolution,
    get_temp_dir,
    is_image,
    is_video,
    normalize_output_path,
)


def show_output_file():
    normed_output_path = normalize_output_path(globals.target_path, globals.output_path)
    if os.path.exists(normed_output_path):
        if is_image(normed_output_path):
            image = cv2.imread(normed_output_path)
            window_width = 1000  # 你可以根据需要设置这个值
            window_height = 1000  # 你可以根据需要设置这个值
            # 计算图像的纵横比
            (h, w) = image.shape[:2]
            ratio = w / h
            # 根据窗口尺寸和纵横比计算图像的新尺寸
            if w > h:
                new_w = window_width
                new_h = int(window_width / ratio)
            else:
                new_h = window_height
                new_w = int(window_height * ratio)

            # 缩放图像
            resized_image = cv2.resize(image, (new_w, new_h))
            cv2.imshow("image", resized_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if is_video(normed_output_path):
            cap = cv2.VideoCapture(normed_output_path)
            while True:
                ret, frame = cap.read()  # 从视频文件读取一帧
                if not ret:
                    print("Reached end of the video or failed to grab frame")
                    break
                cv2.imshow("Video Playback", frame)  # 展示帧

                # 按 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


def test_conditional_process() -> None:
    # "face_debugger", "face_swapper", "frame_enhancer", "frame_colorizer"
    globals.frame_processor = "frame_colorizer"
    globals.target_path = os.path.abspath(r"input\tsq_short.mp4")
    globals.source_path = [os.path.abspath(r"input\glt3.png")]

    globals.output_path = os.path.abspath(r"output")
    globals.reference_frame_number = 0
    reference_frame = os.path.abspath(r"input\0142.png")
    conditional_append_reference_faces(reference_frame)
    conditional_process()
    show_output_file()


def short_video() -> None:
    process_manager.start()
    globals.target_path = os.path.abspath(r"input\target-240p.mp4")
    globals.source_path = [os.path.abspath(r"input\glt4.png")]
    globals.output_path = os.path.abspath(r"output")
    if extract_frames(globals.target_path):
        temp_path = get_temp_dir(globals.target_path)
        image_paths = [os.path.join(temp_path, item) for item in os.listdir(temp_path)]
        for image_path in image_paths[2:]:
            os.remove(image_path)
        resolution = detect_video_resolution(globals.target_path)
        resolution = str(resolution[0]) + "x" + str(resolution[1])
        fps = detect_video_fps(globals.target_path)
        if merge_video(globals.target_path, resolution, fps):
            pass
            # if restore_audio(globals.target_path, globals.output_path):
            #     print("all done")
            # else:
            #     print("restore audio faile")
        else:
            print("merge video failed")
    else:
        print("extract frames failed")
    clear_temp(globals.target_path)


# short_video()
test_conditional_process()
