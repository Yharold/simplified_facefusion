import time
from typing import Tuple
from front_end.core import get_ui_component
import gradio as gr
from mid_end import process_manager, globals
from mid_end.core import conditional_process
from mid_end.vision import (
    clear_temp,
    get_output_path,
    is_image,
    is_video,
    normalize_output_path,
)

START_BUTTON: gr.Button = None
STOP_BUTTON: gr.Button = None
CLEAR_BUTTON: gr.Button = None


def render() -> None:
    global START_BUTTON, STOP_BUTTON, CLEAR_BUTTON
    START_BUTTON = gr.Button("开始执行", visible=True, variant="primary", size="sm")
    STOP_BUTTON = gr.Button("停止执行", visible=True, variant="primary", size="sm")
    CLEAR_BUTTON = gr.Button("清理缓存", visible=True, variant="primary", size="sm")


def listen() -> None:
    output_image = get_ui_component("output_image")
    output_video = get_ui_component("output_video")
    START_BUTTON.click(update, outputs=[STOP_BUTTON, CLEAR_BUTTON])
    START_BUTTON.click(start_process, outputs=[output_image, output_video])
    STOP_BUTTON.click(stop_process, outputs=[START_BUTTON, STOP_BUTTON])
    CLEAR_BUTTON.click(clear_process, outputs=[output_image, output_video])


def update() -> Tuple[gr.Button, gr.Button]:
    while not process_manager.is_processing():
        time.sleep(0.5)
    return gr.Button(visible=True), gr.Button(visible=True)


def start_process() -> Tuple[gr.Image, gr.Video]:
    globals.output_path = get_output_path()
    normed_output_path = normalize_output_path(globals.target_path, globals.output_path)
    conditional_process()
    if is_image(normed_output_path):
        return gr.Image(value=normed_output_path, visible=True), gr.Video(
            value=None, visible=False
        )
    if is_video(normed_output_path):
        return gr.Image(value=None, visible=False), gr.Video(
            value=normed_output_path, visible=True
        )
    return gr.Image(value=None), gr.Video(value=None)


def stop_process() -> Tuple[gr.Button, gr.Button]:
    process_manager.stop()
    return gr.Button(visible=True), gr.Button(visible=False)


def clear_process() -> Tuple[gr.Image, gr.Video]:
    while process_manager.is_processing():
        time.sleep(0.5)
    if globals.target_path:
        clear_temp(globals.target_path)
    return gr.Image(value=None), gr.Video(value=None)
