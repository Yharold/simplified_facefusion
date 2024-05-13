import os
import subprocess
from typing import List, Tuple
from mid_end import process_manager, logger
from mid_end.typing import Fps
from mid_end.vision import detect_video_fps, detect_video_resolution, get_temp_dir

ffmpeg_path = r"ffmpeg\ffmpeg.exe"


def get_abs_ffmege_path() -> str:
    return os.path.abspath(ffmpeg_path)


def run_ffmpeg(args: List[str]) -> bool:
    commands = [get_abs_ffmege_path(), "-hide_banner", "-loglevel", "error"]
    commands.extend(args)
    process = subprocess.Popen(commands, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    while process_manager.is_processing():
        try:
            _, stderr = process.communicate()
            errors = stderr.decode().split(os.linesep)
            for error in errors:
                if error.strip():
                    logger.debug(error.strip(), __name__.upper())
            return process.wait(timeout=0.5) == 0
        except subprocess.TimeoutExpired:
            continue
    return process.returncode == 0


def extract_frames(target_path: str) -> bool:
    temp_path = get_temp_dir(target_path)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    temp_frames_pattern = os.path.join(temp_path, "%04d" + ".png")
    commands = ["-hwaccel", "auto", "-i", target_path, "-q:v", "2"]
    temp_video_resolution = detect_video_resolution(target_path)
    temp_video_fps = detect_video_fps(target_path)
    temp_video_resolution = f"{temp_video_resolution[0]}x{temp_video_resolution[1]}"
    commands.extend(
        ["-vf", "scale=" + temp_video_resolution + ",fps=" + str(temp_video_fps)]
    )
    commands.extend(["-vsync", "0", temp_frames_pattern])
    return run_ffmpeg(commands)


def merge_video(
    target_path: str, output_video_resolution: str, output_video_fps: Fps
) -> bool:
    temp_path = get_temp_dir(target_path)
    temp_frames_pattern = os.path.join(temp_path, "%04d" + ".png")
    temp_output_video_path = os.path.join(temp_path, "temp.mp4")
    commands = [
        "-hwaccel",
        "auto",
        "-s",
        str(output_video_resolution),
        "-r",
        str(output_video_fps),
        "-i",
        temp_frames_pattern,
        "-c:v",
        "libx264",
    ]
    output_video_compression = round(51 - (80 * 0.51))
    commands.extend(
        [
            "-crf",
            str(output_video_compression),
            "-preset",
            "veryfast",
        ]
    )
    commands.extend(
        [
            "-vf",
            "framerate=fps=" + str(output_video_fps),
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-y",
            temp_output_video_path,
        ]
    )
    return run_ffmpeg(commands)

def restore_audio(target_path : str, output_path : str) -> bool:
    # E:\Code\simplified_facefusion\temp\tsq\temp.mp4
    temp_path = get_temp_dir(target_path)
    temp_output_video_path = os.path.join(temp_path, "temp.mp4")
    commands = [ '-hwaccel', 'auto', '-i', temp_output_video_path ]
    commands.extend([ '-i', target_path, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path ])
    return run_ffmpeg(commands)