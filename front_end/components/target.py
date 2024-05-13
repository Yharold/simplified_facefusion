from typing import Tuple
from front_end.core import register_ui_component
import gradio as gr
from mid_end.typing import File
from mid_end.vision import is_image, is_video
from mid_end import globals

TARGET_FILE: gr.File = None
TARGET_IMAGE: gr.Image = None
TARGET_VIDEO: gr.Video = None


def render() -> None:
    global TARGET_FILE, TARGET_IMAGE, TARGET_VIDEO
    TARGET_FILE = gr.File(
        file_count="single",
        file_types=[".png", ".jpg", "mp4"],
        value=None,
        visible=True,
    )
    TARGET_IMAGE = gr.Image(show_label=False, visible=False)
    TARGET_VIDEO = gr.Video(show_label=False, visible=False)
    register_ui_component("target_image", TARGET_IMAGE)
    register_ui_component("target_video", TARGET_VIDEO)


def listen() -> None:
    TARGET_FILE.change(update, inputs=TARGET_FILE, outputs=[TARGET_IMAGE, TARGET_VIDEO])


def update(file: File) -> Tuple[gr.Image, gr.Video]:
    if file:
        if is_image(file.name):
            globals.target_path = file.name
            return (
                gr.Image(show_label=False, visible=True, value=file.name),
                gr.Video(show_label=False, visible=False),
            )
        if is_video(file.name):
            globals.target_path = file.name
            globals.reference_frame_number = 0
            return (
                gr.Image(show_label=False, visible=False),
                gr.Video(show_label=False, visible=True, value=file.name),
            )
    globals.target_path = None
    return (
        gr.Image(show_label=False, visible=False, value=None),
        gr.Video(show_label=False, visible=False, value=None),
    )
