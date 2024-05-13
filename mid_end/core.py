import os
import time
from back_end.core import load_frame_processor_module
from back_end.face_analyser import get_many_faces
from mid_end import logger, globals
from mid_end.face_store import append_reference_faces, get_reference_faces
from mid_end.ffmpeg import (
    extract_frames,
    get_abs_ffmege_path,
    merge_video,
    restore_audio,
)
from mid_end.models import MODELS
from mid_end.vision import (
    clear_temp,
    detect_video_fps,
    detect_video_resolution,
    get_temp_dir,
    is_image,
    is_video,
    is_file,
    normalize_output_path,
    read_static_image,
)
from mid_end import process_manager


# 检查模型是否存在
def check_models() -> bool:
    for item in MODELS.values():
        path = item.get("path")
        if not is_file(path):
            return False
    return True


def check_ffmpeg() -> bool:
    return is_file(get_abs_ffmege_path())


# 添加参考人物face
def conditional_append_reference_faces(frame_path: str) -> None:
    temp_frame = read_static_image(frame_path)
    face = get_many_faces(temp_frame)
    if face:
        append_reference_faces("reference", face[0])


def run() -> None:
    pass


def conditional_process() -> None:
    start_time = time.time()
    logger.init("info")
    logger.enable()
    if not check_models():
        logger.error(
            "some models don't exist! please donwload models!", __name__.upper()
        )
    if not check_ffmpeg():
        logger.error("ffmpeg don't exist! please donwload", __name__.upper())
    if is_image(globals.target_path):
        process_image(start_time)
    if is_video(globals.target_path):
        process_video(start_time)


def process_image(start_time: float) -> None:
    # 判断条件是否满足
    if not check_process_requirements():
        return
    # 输出路径
    normed_output_path = normalize_output_path(globals.target_path, globals.output_path)
    # 分析图片，不是什么不良内容

    # 设置状态
    process_manager.start()
    # 处理
    frame_processor_module = load_frame_processor_module(globals.frame_processor)
    logger.info("Processing", frame_processor_module.__name__.upper())
    frame_processor_module.process_image(
        globals.source_path, globals.target_path, normed_output_path
    )
    frame_processor_module.post_process()
    # 判断是否停止
    if is_process_stopping():
        return
    # 设置状态
    process_manager.end()
    logger.info(
        f"process time is {time.time()-start_time:.2f} seconds", __name__.upper()
    )


def check_process_requirements() -> bool:
    # 无论哪种方法，目标文件和输出路径都必须存在
    if (
        is_image(globals.target_path)
        or is_video(globals.target_path)
        and globals.output_path
    ):
        # 如果是AI换脸，那么要换的人脸和被换的人脸都要确定
        if globals.frame_processor == "face_swapper":
            flag = [is_image(item) for item in globals.source_path]
            return any(flag) and get_reference_faces()
        else:
            return True
    return False


def process_video(start_time: float) -> None:
    if not check_process_requirements():
        return
    normed_output_path = normalize_output_path(globals.target_path, globals.output_path)
    # 设置状态
    process_manager.start()
    if extract_frames(globals.target_path):
        logger.info("extract video success!", __name__.upper())
    else:
        if is_process_stopping():
            return
        return
    temp_path = get_temp_dir(globals.target_path)
    if temp_path:
        temp_frame_paths = [
            os.path.join(temp_path, item)
            for item in os.listdir(temp_path)
            if item.endswith((".png"))
        ]
        frame_processor_module = load_frame_processor_module(globals.frame_processor)
        logger.info("Processing", frame_processor_module.__name__.upper())
        frame_processor_module.process_video(globals.source_path, temp_frame_paths)
        frame_processor_module.post_process()
        # 判断是否停止
        if is_process_stopping():
            return
    else:
        logger.error("temp frame paths not found", __name__.upper())
        return
    resolution = detect_video_resolution(globals.target_path)
    resolution = str(resolution[0]) + "x" + str(resolution[1])
    fps = detect_video_fps(globals.target_path)
    if merge_video(globals.target_path, resolution, fps):
        logger.info("merge video succeed", __name__.upper())
    else:
        logger.error("merge video failed", __name__.upper())
    if restore_audio(globals.target_path, normed_output_path):
        logger.info("restore audio succeed", __name__.upper())
    else:
        logger.error("restore audio failed", __name__.upper())
    # 设置状态
    process_manager.end()
    # 清理临时目录
    clear_temp(globals.target_path)
    # 打印用时
    logger.info(
        f"process time is {time.time()-start_time:.2f} seconds", __name__.upper()
    )


def is_process_stopping() -> bool:
    if process_manager.is_stopping():
        process_manager.end()
        logger.info("Processing stopped", __name__.upper())
    return process_manager.is_pending()
