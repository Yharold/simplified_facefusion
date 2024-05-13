from functools import lru_cache
import subprocess
from typing import List
import xml.etree.ElementTree as ElementTree
from mid_end.typing import ExecutionDevice, ValueAndUnit
import onnxruntime

execution_thread_count = 4
execution_queue_count = 1
frame_processors = ["face_debugger", "face_swapper", "face_enhance", "face_recolor"]


def get_providers():
    providers = []
    available_execution_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_execution_providers:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": (
                        "EXHAUSTIVE" if use_exhaustive() else "DEFAULT"
                    )
                },
            )
        )
    else:
        providers.append("CPUExecutionProvider")
    return providers


def use_exhaustive() -> bool:
    execution_devices = detect_static_execution_devices()
    product_names = ("GeForce GTX 1630", "GeForce GTX 1650", "GeForce GTX 1660")
    return any(
        execution_device.get("product").get("name").startswith(product_names)
        for execution_device in execution_devices
    )


@lru_cache(maxsize=None)
def detect_static_execution_devices() -> List[ExecutionDevice]:
    return detect_execution_devices()


def detect_execution_devices() -> List[ExecutionDevice]:
    execution_devices: List[ExecutionDevice] = []
    try:
        output, _ = run_nvidia_smi().communicate()
        root_element = ElementTree.fromstring(output)
    except Exception:
        root_element = ElementTree.fromstring("xml")
    for gup_element in root_element.findall("gpu"):
        execution_devices.append(
            {
                "driver_version": root_element.find("driver_version").text,
                "framework": {
                    "name": "CUDA",
                    "version": root_element.find("cuda_version").text,
                },
                "product": {
                    "vendor": "NVIDIA",
                    "name": gup_element.find("product_name").text.replace(
                        "NVIDIA", " "
                    ),
                    "architecture": gup_element.find("product_architecture").text,
                },
                "video_memory": {
                    "total": create_value_and_unit(
                        gup_element.find("fb_memory_usage/total").text
                    ),
                    "free": create_value_and_unit(
                        gup_element.find("fb_memory_usage/free").text
                    ),
                },
                "ExecutionDeviceUtilization": {
                    "gpu": create_value_and_unit(
                        gup_element.find("utilization/gpu_util").text
                    ),
                    "memory": create_value_and_unit(
                        gup_element.find("utilization/memory_util").text
                    ),
                },
            }
        )
    return execution_devices


def create_value_and_unit(text: str) -> ValueAndUnit:
    value, unit = text.split()
    value_and_unit: ValueAndUnit = {
        "value": value,
        "unit": unit,
    }
    return value_and_unit


def run_nvidia_smi() -> subprocess.Popen[bytes]:
    commands = ["nvidia-smi", "--query", "--xml-format"]
    return subprocess.Popen(commands, stdout=subprocess.PIPE)
