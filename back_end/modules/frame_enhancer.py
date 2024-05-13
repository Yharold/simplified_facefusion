from mid_end.execution import get_providers
from mid_end.models import MODELS
from mid_end.typing import (
    FrameEnhancerInputs,
    QueuePayload,
    UpdateProgress,
    VisionFrame,
)
import threading, time, onnxruntime, numpy, cv2
from typing import Any, List
from back_end.face_analyser import clear_face_analyser
from mid_end.vision import (
    create_tile_frames,
    merge_tile_frames,
    read_image,
    read_static_image,
    write_image,
)
from mid_end import process_manager
from back_end import core

"""
这个文件的功能是高清放大
"""
FRAME_PROCESSOR = None
THREAD_LOCK: threading.Lock = threading.Lock()


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FRAME_PROCESSOR is None:
            FRAME_PROCESSOR = onnxruntime.InferenceSession(
                MODELS.get("real_esrgan").get("path"),
                providers=get_providers(),
            )
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    FRAME_PROCESSOR = None


def pre_process() -> bool:
    return True


def post_process() -> None:
    read_static_image.cache_clear()
    clear_frame_processor()
    clear_face_analyser()


def get_reference_frame():
    pass


def process_frame(inputs: FrameEnhancerInputs) -> VisionFrame:
    target_vision_frame = inputs.get("target_vision_frame")
    return enhance_frame(target_vision_frame)


def process_frames(
    source_paths: List[str],
    queue_payloads: List[QueuePayload],
    update_progress: UpdateProgress,
) -> None:
    for queue_payload in process_manager.manage(queue_payloads):
        target_vision_path = queue_payload["frame_path"]
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(target_vision_path, output_vision_frame)
        update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame({"target_vision_frame": target_vision_frame})
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    core.multi_process_frames(None, temp_frame_paths, process_frames)


def enhance_frame(temp_vision_frame: VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    size = MODELS.get("real_esrgan").get("size")
    scale = MODELS.get("real_esrgan").get("scale")
    temp_height, temp_width = temp_vision_frame.shape[:2]
    tile_vision_frames, pad_width, pad_height = create_tile_frames(
        temp_vision_frame, size
    )
    # 预处理,运行模型
    for index, tile_vision_frame in enumerate(tile_vision_frames):
        tile_vision_frame = frame_processor.run(
            None,
            {
                frame_processor.get_inputs()[0].name: prepare_tile_frame(
                    tile_vision_frame
                )
            },
        )[0]
        # 后处理
        tile_vision_frames[index] = normalize_tile_frame(tile_vision_frame)
    # 合并tiles，放大倍数是scale，所以图像大小应该是原始大小乘scale
    merge_vision_frame = merge_tile_frames(
        tile_vision_frames,
        temp_width * scale,
        temp_height * scale,
        pad_width * scale,
        pad_height * scale,
        (size[0] * scale, size[1] * scale, size[2] * scale),
    )
    # 混合
    frame_enhancer_blend = 1 - (80.0 / 100)
    temp_vision_frame = cv2.resize(
        temp_vision_frame, (merge_vision_frame.shape[1], merge_vision_frame.shape[0])
    )
    temp_vision_frame = cv2.addWeighted(
        temp_vision_frame,
        frame_enhancer_blend,
        merge_vision_frame,
        1 - frame_enhancer_blend,
        0,
    )
    return temp_vision_frame


def prepare_tile_frame(vision_tile_frame: VisionFrame) -> VisionFrame:
    vision_tile_frame = numpy.expand_dims(vision_tile_frame[:, :, ::-1], axis=0)
    vision_tile_frame = vision_tile_frame.transpose(0, 3, 1, 2)
    vision_tile_frame = vision_tile_frame.astype(numpy.float32) / 255
    return vision_tile_frame


def normalize_tile_frame(vision_tile_frame: VisionFrame) -> VisionFrame:
    vision_tile_frame = vision_tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
    vision_tile_frame = vision_tile_frame.clip(0, 255).astype(numpy.uint8)[:, :, ::-1]
    return vision_tile_frame
