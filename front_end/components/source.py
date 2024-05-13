from typing import Any, List, Optional
from front_end.core import get_ui_component, register_ui_component
import gradio as gr
from mid_end.vision import is_image
from front_end import choices_value
from mid_end import globals

SOURCE_FILE: gr.File = None
SOURCE_IMAGE: gr.Image = None


def render() -> None:
    global SOURCE_FILE, SOURCE_IMAGE, SOURCE_VIDEO
    SOURCE_FILE = gr.File(
        file_count="multiple",
        file_types=[".png", ".jpg"],
        value=None,
        visible=False,
    )
    SOURCE_IMAGE = gr.Image(show_label=False, visible=False)
    register_ui_component("source_image", SOURCE_IMAGE)


def listen() -> None:
    # gr.File设置visible同样会出发change，而且value是None
    SOURCE_FILE.change(update, inputs=SOURCE_FILE, outputs=SOURCE_IMAGE)
    process_tool = get_ui_component("process_tool")
    process_tool.change(update_source_file, inputs=process_tool, outputs=SOURCE_FILE)


def update(files: List[Any]) -> gr.Image:
    file_names = [file.name for file in files if is_image(file.name)] if files else None
    if file_names:
        globals.source_path = file_names
        return gr.Image(show_label=False, visible=True, value=file_names[0])
    return gr.Image(show_label=False, visible=False, value=None)


def update_source_file(choice: str) -> gr.File:
    if choices_value.tool_choices.index(choice) == 1:
        return gr.File(
            show_label=False,
            file_count="multiple",
            file_types=[".png", ".jpg"],
            value=None,
            visible=True,
        )
    return gr.File(
        show_label=False,
        file_count="multiple",
        file_types=[".png", ".jpg"],
        value=None,
        visible=False,
    )
