import hashlib
import os
import shutil
import filetype
from functools import lru_cache
from typing import List, Optional, Tuple
import cv2
from mid_end.typing import Resolution, VisionFrame
from cv2.typing import Size
import numpy
from mid_end import globals

"""
这个文件是处理图像和路径的

"""
temp_directory = r"temp"
output_directory = r"output"


def get_temp_path():
    temp_path = os.path.abspath(temp_directory)
    if os.path.exists(temp_path):
        return temp_path
    os.makedirs(temp_path)
    return temp_path


def get_output_path():
    temp_path = os.path.abspath(output_directory)
    if os.path.exists(temp_path):
        return temp_path
    os.makedirs(temp_path)
    return temp_path


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path: str) -> bool:
    return bool(directory_path and os.path.isdir(directory_path))


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def has_image(image_paths: List[str]) -> bool:
    if image_paths:
        return any(is_image(image_path) for image_path in image_paths)
    return False


def is_video(video_path: str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


@lru_cache(maxsize=128)
def read_static_image(image_path: str) -> Optional[VisionFrame]:
    return read_image(image_path)


def read_static_images(image_paths: List[str]) -> Optional[List[VisionFrame]]:
    frames = []
    if image_paths:
        for image_path in image_paths:
            frames.append(read_static_image(image_path))
    return frames


def read_image(image_path: str) -> VisionFrame:
    if is_image(image_path):
        return cv2.imread(image_path)
    return None


def resize_frame_resolution(
    vision_frame: VisionFrame, resolution: Resolution
) -> VisionFrame:
    height, width = vision_frame.shape[:2]
    max_width, max_height = resolution
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(vision_frame, (new_width, new_height))
    return vision_frame


def write_image(image_path: str, vision_frame: VisionFrame) -> bool:
    if image_path:
        return cv2.imwrite(image_path, vision_frame)
    return False


def detect_video_resolution(video_path: str) -> Optional[Resolution]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_capture.release()
            return int(width), int(height)
    return None


def detect_video_fps(video_path: str) -> Optional[float]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_capture.release()
            return video_fps
    return None


def get_temp_dir(target_path: str) -> str:
    temp_dir = get_temp_path()
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(temp_dir, target_name)


def normalize_output_path(
    target_path: Optional[str], output_path: Optional[str]
) -> Optional[str]:
    if target_path and output_path:
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if is_directory(output_path):
            output_hash = hashlib.sha1(
                str(globals.__dict__).encode("utf-8")
            ).hexdigest()[:8]
            output_name = target_name + "-" + output_hash
            return os.path.join(output_path, output_name + target_extension)
        output_name, output_extension = os.path.splitext(os.path.basename(output_path))
        output_directory_path = os.path.dirname(output_path)
        if is_directory(output_directory_path) and output_extension:
            return os.path.join(output_directory_path, output_name + target_extension)
    return None


def clear_temp(target_path: str) -> None:
    temp_directory_path = get_temp_dir(target_path)
    if is_directory(temp_directory_path):
        shutil.rmtree(temp_directory_path, ignore_errors=True)


# 假设size = (128,8,2)
# 将图片划分为大小是128x128的tile，8是大的边界填充的宽度，2是每个tile重合的宽度
# 所以图片实际大小应该是124x124，只不过上下左右都重合了2个单位，这样得到的tile就是128x128
def create_tile_frames(
    vision_frame: VisionFrame, size: Size
) -> Tuple[List[VisionFrame], int, int]:
    # 假设vision_frame原本大小是(598,620,3)，这是随便给的数字,这一步执行完后就是(614,636,3)
    vision_frame = numpy.pad(
        vision_frame, ((size[1], size[1]), (size[1], size[1]), (0, 0))
    )
    # 不重合的tile大小是s0-2*s2,这里就是128-2*2=124
    tile_width = size[0] - 2 * size[2]
    # 再计算还需要填充的像素数量
    # 以高为例，vision_frame.shape[0] % tile_width 得到的就是最后一个tile的高
    # tile_width就是tile_height,所以tile_width+s2-目前的像素数得到的就是还需要填充的像素数
    # 614%124=118，此时pad_size_bottom=2+124-118=8
    # 636%124=16，此时pad_size_right = 110
    pad_size_bottom = size[2] + tile_width - vision_frame.shape[0] % tile_width
    pad_size_right = size[2] + tile_width - vision_frame.shape[1] % tile_width
    # 这一步执行后，vision_frame就从(614,636,3)变成了(624,748,3)
    # 624-2*2=620,620÷124=5；748-2*2=744,744÷124=6,刚好除尽
    pad_vision_frame = numpy.pad(
        vision_frame, ((size[2], pad_size_bottom), (size[2], pad_size_right), (0, 0))
    )
    # 扩充后的高和宽
    pad_height, pad_width = pad_vision_frame.shape[:2]
    row_range = range(size[2], pad_height - size[2], tile_width)
    col_range = range(size[2], pad_width - size[2], tile_width)
    tile_vision_frames = []
    # 行
    for row_vision_frame in row_range:
        # 行的顶和底像素索引
        top = row_vision_frame - size[2]
        bottom = row_vision_frame + size[2] + tile_width
        # 列
        for column_vision_frame in col_range:
            # 列的左和右像素索引
            left = column_vision_frame - size[2]
            right = column_vision_frame + size[2] + tile_width
            tile_vision_frames.append(pad_vision_frame[top:bottom, left:right, :])
    return tile_vision_frames, pad_width, pad_height


def merge_tile_frames(
    tile_vision_frames: List[VisionFrame],
    temp_width: int,
    temp_height: int,
    pad_width: int,
    pad_height: int,
    size: Size,
) -> VisionFrame:
    # 合并后的图像大小为(pad_height, pad_width, 3)，这里先创建一个空白的图像
    # 假设(pad_height, pad_width, 3)：(2496,2992,3),这是原始图像带扩边的放大后的大小
    # 假设(temp_height, temp_width, 3)：(2392,2480,3)，这是原始图像放大后的大小
    merge_vision_frame = numpy.zeros((pad_height, pad_width, 3)).astype(numpy.uint8)
    # 每一个tile的大小是(tile_width,tile_width)，宽高都相同
    tile_width = tile_vision_frames[0].shape[1] - 2 * size[2]
    # 一行有几个tile
    tiles_per_row = min(pad_width // tile_width, len(tile_vision_frames))

    for index, tile_vision_frame in enumerate(tile_vision_frames):
        # 将重合边界去除，只保留不重合部分
        tile_vision_frame = tile_vision_frame[size[2] : -size[2], size[2] : -size[2]]
        # 行号
        row_index = index // tiles_per_row
        # 列号
        col_index = index % tiles_per_row
        # 计算tile在整个图像中的上下左右位置
        top = row_index * tile_vision_frame.shape[0]
        bottom = top + tile_vision_frame.shape[0]
        left = col_index * tile_vision_frame.shape[1]
        right = left + tile_vision_frame.shape[1]
        merge_vision_frame[top:bottom, left:right, :] = tile_vision_frame
    # 还有一开始扩大的s1边界，需要将这个边界也去除
    merge_vision_frame = merge_vision_frame[
        size[1] : size[1] + temp_height, size[1] : size[1] + temp_width, :
    ]
    return merge_vision_frame


def count_video_frame_number(video_path) -> int:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            video_frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()
            return video_frame_number
    return 0


def get_temp_frame(video_path: str, frame_number: int) -> Optional[VisionFrame]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_capture.set(
                cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1)
            )
            has_vision_frame, vision_frame = video_capture.read()
            video_capture.release()
            if has_vision_frame:
                return vision_frame
    return None


def normalize_frame_color(vision_frame: VisionFrame) -> VisionFrame:
    return cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)
