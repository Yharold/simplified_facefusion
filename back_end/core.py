from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import importlib, sys, os
from types import ModuleType
from typing import Any, List

from mid_end.execution import (
    get_providers,
    execution_thread_count,
    execution_queue_count,
    frame_processors,
)
from tqdm import tqdm

from mid_end.typing import ProcessFrames, QueuePayload
from mid_end import logger


FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_METHODS = [
    "get_frame_processor",
    "clear_frame_processor",
    "pre_process",
    "post_process",
    "get_reference_frame",
    "process_frame",
    "process_frames",
    "process_image",
    "process_video",
]


# 载入模块
def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(
            "back_end.modules." + frame_processor
        )
        for method_name in FRAME_PROCESSORS_METHODS:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplemented
    except ModuleNotFoundError as exception:
        logger.error(
            "Frame processor {frame_processor} could not be loaded".format(
                frame_processor=frame_processor
            ),
            __name__.upper(),
        )
        logger.debug(exception.msg, __name__.upper())
        sys.exit(1)
    except NotImplementedError:
        logger.error(
            "Frame processor {frame_processor} not implemented correctly".format(
                frame_processor=frame_processor
            ),
            __name__.upper(),
        )
        sys.exit(1)
    return frame_processor_module


# 得到模块
def get_frame_processor_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES
    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


# 执行模块
def multi_process_frames(
    source_paths: List[str], temp_frame_paths: List[str], processe_frames: ProcessFrames
) -> None:
    queue_payloads = create_queue_payloads(temp_frame_paths)
    with tqdm(
        total=len(queue_payloads),
        desc="Processing",
        unit="frame",
        ascii=" =",
        disable=False,
    ) as progress:
        progress.set_postfix(
            {
                "execution_providers": get_providers()[0][0],
                "execution_thread_count": execution_thread_count,
                "execution_queue_count": execution_queue_count,
            }
        )
        with ThreadPoolExecutor(max_workers=execution_thread_count) as executor:
            futures = []
            queue: Queue[QueuePayload] = create_queue(queue_payloads)
            queue_per_future = max(
                len(queue_payloads) // execution_thread_count * execution_queue_count, 1
            )
            while not queue.empty():
                future = executor.submit(
                    processe_frames,
                    source_paths,
                    pick_queue(queue, queue_per_future),
                    progress.update,
                )
                futures.append(future)
            for future_done in as_completed(futures):
                future_done.result()


# 清理模块
def clear_processors_modules() -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor_module in get_frame_processor_modules(frame_processors):
        frame_processor_module.clear_frame_processor()
    FRAME_PROCESSORS_MODULES = []


def create_queue_payloads(temp_frame_paths: List[str]) -> List[QueuePayload]:
    queue_payloads = []
    temp_frame_paths = sorted(temp_frame_paths, key=os.path.basename)
    for frame_number, frame_path in enumerate(temp_frame_paths):
        frame_payload: QueuePayload = {
            "frame_number": frame_number,
            "frame_path": frame_path,
        }
        queue_payloads.append(frame_payload)
    return queue_payloads


def create_queue(queue_payloads: List[QueuePayload]) -> Queue[QueuePayload]:
    queue: Queue[QueuePayload] = Queue()
    for queue_payload in queue_payloads:
        queue.put(queue_payload)
    return queue


def pick_queue(queue: Queue[QueuePayload], queue_per_future: int) -> List[QueuePayload]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues
